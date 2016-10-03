import numpy as np

class pla:    
    def __init__(self, X, y, max_iterations = np.inf):
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]
        self.X = np.insert(X, 0, np.ones(self.n_examples), axis=1)
        self.y = y
        self.w = np.zeros(self.n_features + 1)
        if max_iterations:
            self.max_iterations = max_iterations

    def fit(self):
        # reset the weight to zeros:
        self.w = np.zeros(self.n_features + 1)

        # cycle through patterns:
        self.n_iterations = 0
        while(self.n_iterations < self.max_iterations):
            self.n_iterations += 1
            for i in range(self.n_examples):
				# predict:
                y_pred = np.sign(np.dot(self.X[i,:], self.w))
                if not y_pred == self.y[i]:
                    self.w = self.w + self.y[i] * self.X[i,:]
                    break
            else:
                break
    
    def predict(self, x):
        if x.shape[1] != self.n_features + 1:
            x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        return np.sign(np.dot(x, self.w))
