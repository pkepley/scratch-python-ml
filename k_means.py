# testing out k-means
import numpy as np
from scipy.spatial import KDTree

import matplotlib.pyplot as plt


class k_means:

	def __init__(self, X, k):
		self.X = X
		self.k = k
		# cluster every training example according to Mu
		self.n_examples = X.shape[0]
		self.n_features = X.shape[1]


	def random_initializer(self):		
		# select k indices randomly without replacement
		ii = np.random.choice(self.n_examples, self.k, replace = False)	
		
		# use the selected entries as initializers
		Mu = self.X[ii,:]
		return Mu

	def cluster_points(self, Mu):	
		# find the nearest entry in Mu to each training example
		tree = KDTree(Mu)
		neighbor, classifier = tree.query(self.X)		
		return classifier

	def cluster_means(self, c):
		Mu = np.zeros((self.k, self.n_features))
		for i in range(0,self.k):
			Mu[i,:] = np.mean(self.X[c == i],0)
		return Mu

	def J(self, Mu, c):
		# nearest_centroid = Mu[c]
		JJ = 0.0
		for i in range(0,self.k):
			JJ += np.linalg.norm(self.X[c == i] - Mu[i])**2
		return JJ
	
	def fit(self, n_outer = 1):
		# X       - should be input as an array. Each row should 
		#           correspond to a training example.
		# k       - denotes the number of training classes
		# n_outer - the number of times we repeat k means
		
		J_min = np.inf
		
		for i in range(0, n_outer):
			# generate centroids randomly
			Mu = self.random_initializer()
			Mu_old = Mu
			
			converged = False		
			while(not converged):
				# cluster according to nearest neighbor
				c  = self.cluster_points(Mu)
				
				# compute the cluster means
				Mu = self.cluster_means(c)
				
				# distance between mus
				d = np.linalg.norm(Mu - Mu_old)
				
				if d > 0:
					Mu_old = Mu
				else:
					converged = True
				
			# compute J
			J_cur = self.J(Mu, c)
			
			if J_cur < J_min:
				J_min = J_cur
				Mu_min = Mu
				c_min = c
				print('Iteration {0}, J was reduced to: {1}'.format(i, J_min))
			
			self.c = c_min
			self.Mu = Mu_min

	def plot_clusters(self):
		if self.n_features == 2:
			plt.gray()
			plt.scatter(self.X[:,0], self.X[:,1], c = self.c, s = 50)
			plt.scatter(self.Mu[:,0], self.Mu[:,1], c = range(self.k), s = 500)
			plt.show()
		else:
			print "Can't plot"
