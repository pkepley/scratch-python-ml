import numpy as np
from scipy.linalg import eig

class lda:
	def __init__(self, X, y):
		self.X = X
		self.n_observations = X.shape[0]
		self.n_features = X.shape[1]

		self.y = y
		self.classes = np.unique(y)
		self.n_classes = len(self.classes)
		if self.n_classes > self.n_features:
			raise ValueError("The number of classes cannot exceed the number of features")

	def fit(self):		
		self.m = np.mean(self.X, axis = 0).reshape((1, self.n_features))
		self.m_k = np.zeros((self.n_classes, self.n_features))
		self.N_k = np.zeros(self.n_classes)
		self.S_w = np.zeros((self.n_features, self.n_features))
		self.S_b = np.zeros((self.n_features, self.n_features))

		for k in range(self.n_classes):
			# membership in class c:
			c_idx = np.where(self.y == self.classes[k])[0]
			
			# number of elements in class c:
			self.N_k[k] = len(c_idx)
			
			# mean of observations in class c:
			self.m_k[k,:] = np.mean(self.X[c_idx,:], axis = 0)

			# within class scatter:
			self.S_w = self.S_w + np.dot((self.X[c_idx,:] - self.m_k[k,:]).T, 
										 (self.X[c_idx,:] - self.m_k[k,:]))
			
			# between class scatter:
			self.S_b = self.S_b + self.N_k[k] * np.dot((self.m_k[k] - self.m).T,
													   self.m_k[k] - self.m)

		# The objective is:
		#
		#  J(W) = trace{ (W S_w W^t)^-1 (W S_b W^T) }
		#
		# According to:
		#   http://www-users.cs.umn.edu/~saad/PDF/umsi-2009-31.pdf
		# Maximizing J can be accomplished by solving the generalized 
		# eigenvalue problem:
		# 
		#  S_b u_i = lambda_i * S_w * u_i
		# 
		self.lambdas, self.W = eig(self.S_b, self.S_w)

		idx = self.lambdas.argsort()[::-1]
		idx = idx[0: self.n_classes]
		self.lambdas = self.lambdas[idx]
		self.W = self.W[:,idx]


	def predict(self, x):
		# project data and class means:
		x_projected = np.dot(self.W.T, x.T)
		m_k_projected = np.dot(self.W.T, self.m_k.T)

		# compute distances to projected means
		d = np.zeros((self.n_classes, x.shape[0]))		
		for k in range(self.n_classes):
			d[k,:] = np.linalg.norm(x_projected - 
									m_k_projected[:,k].reshape((self.n_classes,1)), 
									axis = 0) 
		
		# prediction is the closest projected mean, as discussed in:
		# https://users.cs.fiu.edu/~taoli/pub/Li-discrimant.pdf
		predicted_class_index = np.argmin(d, axis = 0)
		
		# return the class prediction:
		predicted_class = np.apply_along_axis(lambda i: self.classes[i], 0,
											  predicted_class_index)		
		return predicted_class
