import seaborn as sns
import os
import numpy as np

from scipy.stats.mstats import winsorize

import torch

def use_logs(data, *args):
    columns = list(args)
    data = data.loc[:, columns]
    # Winsorize to handle outliers
    data['GR_win'] = winsorize(data['GR'], limits=(0, 0.005))
    data['SP_win'] = winsorize(data['SP'], limits=(0.02, 0.35))
    data['DRHO_win'] = winsorize(data['DRHO'], limits=(0.05, 0.4))
    data.drop(['GR', 'SP', 'DRHO'], axis=1, inplace=True)
    data.fillna(-999, inplace=True)
    return data


A = np.load('data/penalty_matrix.npy')
def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        # Check bounds for safety
        if y_true[i] < A.shape[0] and y_pred[i] < A.shape[1]:
            S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

# Set seeds for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)