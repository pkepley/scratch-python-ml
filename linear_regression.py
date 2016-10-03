import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

class linear_regression:
    def __init__(self, X, y, alpha = 0, tol = 10**-5, eta=10**-6, max_iterations=np.inf):
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]

        # pad X with ones in the zero position:
        self.X = np.insert(X, 0, np.ones(self.n_examples), axis=1)
        self.y = y

        self.alpha = alpha
        self.w = np.zeros(self.n_features + 1)
        self.tol = tol
        self.eta = eta
        self.max_iterations = max_iterations

    def h(self, w, Xs):
        h_w = np.dot(w, Xs.T)
        return h_w.flatten()
    
    def J(self, w, alpha):
        ####################################################
        # without any regularization:
        #    J = (1/2) * average( \|y_w(x_i) - y_i\|^2 )
        #    \nabla J = average( (y_w(X) - y) * X)
        ####################################################
        residual =  self.h(w, self.X) - self.y.flatten()
        J_alpha = (0.5) * np.dot(residual.T, residual) / self.n_examples
        grad_J_alpha = np.dot(residual, self.X) / self.n_examples

        ####################################################
        #  add in the regularizing term: \alpha \|w[1:]\|^2
        #  we do not include the bias term w[0] in the 
        #  regularizing term.
        ####################################################
        J_alpha = J_alpha +  (0.5 * alpha) * ( np.dot(w[1:].T, w[1:]) )
        grad_J_alpha[1:] = grad_J_alpha[1:] + alpha * w[1:]

        return J_alpha, grad_J_alpha

    def fit(self, use_preset_w = False):
        if use_preset_w:
            w = self.w
        else:
            w = np.zeros((1, self.n_features+1))
        J, grad_J = self.J(w, self.alpha)
        n_its = 0
        while(np.dot(grad_J, grad_J) > self.tol and n_its < self.max_iterations):
            J, grad_J = self.J(w, self.alpha)
            w = w - self.eta * grad_J
            n_its += 1
        self.w = w

    def fit_normal_eq(self):
        # solve for w:
        #   (X'X + alpha II) w = X'y
        # where II is the identity but has been
        # zeroed out in the (0,0) entry
        # to avoid regularizing the bias

        # X' * X
        Z = np.dot(self.X.T, self.X)
        for i in range(1, self.n_features + 1):
            Z[i,i] = Z[i,i] + self.alpha

        # X' * y
        v = np.dot(self.X.T, self.y)

        w = solve(Z, v).flatten()
        self.w = w

        
    def pad(self, Xs):
        nxs = Xs.shape[0]
        Xs = np.insert(Xs, 0, np.ones(nxs), axis=1)
        return Xs

    def predict(self, Xs):
        return self.h(self.w, self.pad(Xs))
