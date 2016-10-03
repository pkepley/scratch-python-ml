import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize

class logistic_regression:
    def __init__(self, X, y, alpha = 0, max_iterations = None):
        self.n_observations = X.shape[0]
        self.n_features = X.shape[1]

        # pad X with ones in the zero position:
        self.X = np.insert(X, 0, np.ones(self.n_observations), axis=1)
        self.y = y

        # regularization parameter
        self.alpha = alpha
        
        # weight vector. initially set as 0
        self.w = np.zeros(self.n_features + 1)

        # maximum number of iterations, used for causing
        # minimize to terminate early
        self.max_iterations = max_iterations

    def g(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def h(self, w, Xs):
        h_w = self.g(np.dot(w, Xs.T))
        return h_w.flatten()
    
    def J(self, w):
        J_alpha =  -np.dot(self.y.flatten(), np.log(self.h(w, self.X)))

        # add a very small value inside of the logarithm to avoid numerical zero
        J_alpha -=  np.dot(1.0 - self.y,  np.log(1.0 - self.h(w, self.X) + 10**-30)) 
        J_alpha = J_alpha / self.n_observations

        # add in regularization
        J_alpha = J_alpha +  (0.5 * self.alpha / self.n_observations) * ( np.dot(w[1:].T, w[1:]) )

        return J_alpha

    def grad_J(self, w):
        residual =  self.h(w, self.X) - self.y.flatten()
        grad_J_alpha = np.dot(residual, self.X) / self.n_observations
        grad_J_alpha[1:] = grad_J_alpha[1:] + (self.alpha / self.n_observations) * w[1:]
        return grad_J_alpha

    def fit(self):
        w = np.zeros((1, self.n_features+1))

        if self.max_iterations:
            self.optimize_result = minimize(self.J, w, jac = self.grad_J, options={'maxiter': self.max_iterations})
        else:
            self.optimize_result = minimize(self.J, w, jac = self.grad_J)

        self.w = self.optimize_result.x
        
    def pad(self, Xs):
        nxs = Xs.shape[0]
        Xs = np.insert(Xs, 0, np.ones(nxs), axis=1)
        return Xs

    def predict(self, Xs):
        return self.h(self.w, self.pad(Xs))
        
