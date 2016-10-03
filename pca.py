import numpy as np
from scipy.linalg import svd

class pca:
    def __init__(self, X, n_eigs = None, tol = None):
        self.X = X
        self.n_observations = X.shape[0]
        self.n_features = X.shape[1]
        self.n_eigs = n_eigs
        self.tol = tol        
    
    def fit(self, n_outer = 1):
        self.mean = np.mean(self.X, axis=0);
        self.cov = np.dot((self.X - self.mean).T, (self.X - self.mean));
        # U has singular vectors as columns
        # s has singular values
        U, s, Vh = svd(self.cov)
        self.U = U
        self.s = s

        if self.tol:
            s_sum = np.cumsum(self.s)
            s_tot = s_sum[-1]
            s_sum = s_sum / s_tot
            self.n_eigs = 1 + np.where((1.0 - s_sum) <= self.tol)[0][0]

    def transform(self, x):
        x = x - self.mean
        s_x = np.dot(x, self.U[:, 0:self.n_eigs])
        return s_x

    def inverse_transform(self, s_x):
        x = np.dot( s_x, self.U[:, 0:self.n_eigs].T)
        x = x + self.mean
        return x

    
