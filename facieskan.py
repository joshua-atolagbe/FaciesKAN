import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# KANLinear Layer (B-Spline) 
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).float().unsqueeze(0).expand(in_features, -1)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            x_fit = torch.linspace(-1, 1, steps=self.grid_size + 1).to(self.base_weight.device).unsqueeze(1).expand(-1, self.in_features)
            coeff = self.curve2coeff(x_fit, noise)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * coeff)
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=np.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(2)
        grid = grid.unsqueeze(0)
        
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :, : -(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * bases[:, :, 1:]
        
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).permute(1, 0, 2)
        B = y.permute(1, 0, 2)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous().view(self.out_features, -1)

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        bases = self.b_splines(x) 
        spline_weight_view = self.spline_weight.view(self.out_features, self.in_features, -1)
        spline_part = torch.einsum('big, oig -> bio', bases, spline_weight_view)
        if self.enable_standalone_scale_spline:
            spline_part = spline_part * self.spline_scaler.t()
        spline_output = spline_part.sum(dim=1)
        return base_output + spline_output

# SlefAttention block

class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention: x is (Batch, Seq, Feature)
        # For well logs, we treat the feature vector as a sequence of length 1 or project it
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

# Cross-attention block
class KnowledgeCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, num_knowledge_slots=32, dropout=0.1):
        super().__init__()
        # Learnable Knowledge Base (Memory)
        # This represents "prototypes" of geological features/lithologies
        self.knowledge_base = nn.Parameter(torch.randn(1, num_knowledge_slots, d_model))
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Query = x (Input features)
        # Key/Value = Knowledge Base
        batch_size = x.size(0)
        
        # Expand knowledge base for the batch
        k_base = self.knowledge_base.expand(batch_size, -1, -1)
        
        attn_output, _ = self.cross_attn(query=x, key=k_base, value=k_base)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x
        
class KnowledgeAttentionKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, grid_size=10, 
                 spline_order=3, num_heads=4, knowledge_slots=20, dropout=0.3):
        super(KnowledgeAttentionKAN, self).__init__()
        
        # 1. Multi-scale Feature Extraction (using multiple KAN layers)
        self.feature_extractors = nn.ModuleList([
            KANLinear(input_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order),
            KANLinear(input_dim, hidden_dim, grid_size=grid_size*2, spline_order=spline_order),
        ])
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 2. Batch Normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # 3. Enhanced Self-Attention with residual
        self.self_attention = AttentionBlock(hidden_dim, num_heads, dropout=dropout)
        
        # 4. Cross-Attention with more knowledge slots
        self.knowledge_attention = KnowledgeCrossAttention(
            hidden_dim, num_heads, 
            num_knowledge_slots=knowledge_slots, 
            dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 6. Final classifier with uncertainty modeling
        self.classifier = KANLinear(hidden_dim, output_dim, 
                                   grid_size=grid_size, spline_order=spline_order)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-scale feature extraction
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(x))
        
        # Fusion
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Reshape for attention
        x = x.unsqueeze(1)
        
        # Self-attention
        x_att = self.self_attention(x)
        
        # Cross-attention with knowledge base
        x_know = self.knowledge_attention(x_att)
        
        # Flatten
        x = x_know.squeeze(1)
        
        # Feed-forward with residual
        residual = x
        x = self.ffn(x)
        x = self.bn2(x + residual)
        
        # Classification with temperature scaling
        logits = self.classifier(x)
        
        return logits / self.temperature


# Wrapper with better training practices
class KANClassifierWrapper:
    def __init__(self, input_dim, output_dim, hidden_dim=128, epochs=100, 
                 lr=0.001, batch_size=1024, device=None, grid_size=10, 
                 spline_order=3, patience=15, class_weights=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device if device else ('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.patience = patience
        self.class_weights = class_weights
        
        # Use RobustScaler instead of StandardScaler
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler(quantile_range=(5, 95))
        
        self.model = None
        self.classes_ = np.arange(output_dim)
        self.best_model_state = None

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Initialize model with higher capacity
        # Ensure hidden_dim is divisible by num_heads
        num_heads = 8 if self.hidden_dim % 8 == 0 else 4
        
        self.model = KnowledgeAttentionKAN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            num_heads=num_heads,  # Automatically adjusted
            knowledge_slots=48,  # More knowledge prototypes
            dropout=0.3
        ).to(self.device)
        
        # Use class weights for imbalanced data
        if self.class_weights is not None:
            weights = torch.FloatTensor(self.class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Better optimizer configuration
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=1e-3,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=True, pin_memory=True, num_workers=0
        )
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Training Knowledge Attention KAN on {self.device}...")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            avg_loss = total_loss / len(loader)
            train_acc = correct / total
            scheduler.step()
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_slice = X_scaled[i : i + self.batch_size]
                X_tensor = torch.FloatTensor(batch_slice).to(self.device)
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.cpu())
        
        return torch.cat(predictions).numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        probs_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_slice = X_scaled[i : i + self.batch_size]
                X_tensor = torch.FloatTensor(batch_slice).to(self.device)
                outputs = self.model(X_tensor)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu())
        
        return torch.cat(probs_list).numpy()