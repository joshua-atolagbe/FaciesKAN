import numpy as np

def augment_features_window(X, N_neig):
    N_row = X.shape[0]
    N_feat = X.shape[1]
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row
    return X_aug

def augment_features_gradient(X, depth):
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    return X_grad

def vclay(df):
    list_vclay = []
    for well in df['WELL'].unique():
        df_aux = df[df['WELL'] == well].copy()
        den = (df_aux['GR_win'].max() - df_aux['GR_win'].min())
        if den == 0: den = 1
        df_aux['VCLAY'] = (df_aux['GR_win'] - df_aux['GR_win'].min()) / den
        list_vclay.extend(df_aux['VCLAY'].values)
    df['VCLAY'] = list_vclay
    return df
def augment_features(X, well, depth, N_neig=2):
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    return X_aug, padded_rows