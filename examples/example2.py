import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# this should be the current directory, where the
# examples live:
examples_dir_path = os.path.abspath(os.path.dirname(__file__))

# get logistic regression, should be located in the
# directory one up on the tree:
sys.path.append(examples_dir_path + '/../')
from logistic_regression import logistic_regression

def mapFeature2d(X, degree=6):
	# Polynomial feature mapping for 2d data
	#
	# input: 
	#
	# X     - an array with shape (n_observations,2)
	#         will throw an error if it doesn't have this shape
	# degree - the highest polynomial degree to be considered
	#         this is optional, default is 6
	#
	# output:
	# Xout  - returns an array whose columns consist of all polynomial
	#         products of the features of X with degree greater than 0
	#         and of degree less than the supplied degree 
	#      
	#       ie. looks like:
	#            [x1, x2, x1**2, x1*x2, x2**2,..., x2**degree]

	n_observations = X.shape[0]
	n_features = X.shape[1]
	if n_features != 2:
		raise ValueError("Not 2 dim")

	else:
		# count the number of dimensions:
		ndims = 0
		for j in range(degree + 1):
			for i in range(0,j + 1):
				ndims+=1
				
		# reserve space:
		Xout = np.zeros((n_observations, ndims - 1))

		# map the dimensions:
		dim = 0
		for j in range(1, degree + 1):
			for i in range(0,j + 1):
				Xout[:, dim] = X[:,0]**i * X[:,1]**(j-i)
				dim += 1
		return Xout


def plot_contour(X, y, alpha = 0.0, degree=6, decision_level=0.5, ranges=None):
	# Train Logistic Regression and Plot contour for 2d data using polynomial features
	#
	# input: 
	# X      - an array with shape (n_observations,2)
	#         will throw an error if it doesn't have this shape
	# y      - an array with n_observations
	# alpha  - regularization parameter
	# degree - highest order polynomial degree to use
	# decision_level - threshold to use for the decision boundary
	#                  ie, classify 1 if > decision_level
	#                      and      0 if < decision_level
	# ranges - range for plotting, should be in form [xlo,xhi,ylo,yhi]
	#          this is optional, if not included we will just use data ranges
	
	if ranges:
		xlo,xhi,ylo,yhi = ranges
	else:
		xlo, ylo = np.min(X, axis=0)
		xhi, yhi = np.max(X, axis=0)

	# mesh
	xx,yy = np.meshgrid( np.linspace(xlo, xhi, 100), np.linspace(ylo, yhi, 100))
	pp = np.array([xx.flatten(), yy.flatten()]).T
	pp = mapFeature2d(pp, degree)
	
	# train:
	logr = logistic_regression(mapFeature2d(X, degree), y, alpha=alpha);
	logr.fit()

	# predict 
	zz = logr.predict(pp)
	zz = np.reshape(zz, (100,100))

	# scatter points with seaborn:
	df = pd.DataFrame({'Microchip Test 1' : X[:,0], 
					   'Microchip Test 2' : X[:,1], 
					   'y' : y}); 
	g = sns.FacetGrid(df,  hue='y', size=5, palette="Set3", hue_kws={"marker" : ["o","s"]}, legend_out=False);
	g = (g.map(plt.scatter, 'Microchip Test 1', 'Microchip Test 2', s = 40, linewidth=.5, edgecolor='black')
			  .add_legend());
	g.fig.suptitle(r"Data with Decision Boundary ($\alpha$ = %.02f)" % alpha);
	leg = g.ax.legend(frameon=True)
	leg.get_frame().set_facecolor('w');
	leg.get_frame().set_edgecolor('b');
	leg.get_frame().set_linewidth(1.0);

	# plot contour:
	CS = plt.contour(xx, yy, zz, levels=[ decision_level ], colors=('k',), linewidth=2);
	plt.clabel(CS, inline=1, fontsize=10)
	
	return logr

if __name__ == '__main__':
 	#################################################
	# part 1:
	#################################################

	# read data:
	df = pd.read_csv(examples_dir_path + '/ex2data1.txt', header=None);
	df.rename(index=str, 
			  columns={0 :'Exam 1 Score', 
					   1 : 'Exam 2 Score', 
					   2:'y'}, 
			  inplace=True);

	# extract array from data:
	X = np.array(df); y = X[:,2]; X = X[:,0:2];
	logr = logistic_regression(X,y);
	logr.fit();

	# scatter plot the classes:
	g = sns.FacetGrid(df,  hue='y', size=5, palette="Set3", hue_kws={"marker" : ["o","s"]}, legend_out=False);
	g = (g.map(plt.scatter, 'Exam 1 Score', 'Exam 2 Score', s = 40, linewidth=.5, edgecolor='black')
			  .add_legend());
	g.fig.suptitle("Data with Decision Boundary");
	leg = g.ax.legend(frameon=True)
	leg.get_frame().set_facecolor('w');
	leg.get_frame().set_edgecolor('b');
	leg.get_frame().set_linewidth(1.0);

	#sns.lmplot('Exam 1 Score', 'Exam 2 Score', df, hue='y', fit_reg=False);

	# plot the decision boundary:
	w = logr.w;
	tt = np.linspace(30,100, 2); 
	ss = -(w[0] + w[1] * tt)/w[2];
	plt.plot(tt, ss, linewidth=2, c='k');
	plt.axis([30,100,30,100]);

 	################################################
	# part 2:
	#################################################
	df = pd.read_csv(examples_dir_path + '/ex2data2.txt', header=None);
	X = np.array(df); y = X[:,2]; X = X[:,0:2];
	
	# overfit:
	plot_contour(X, y, alpha = 0.0, degree=6, ranges=None);

	# better:
	plot_contour(X, y, alpha = 1.0, degree=6, ranges=None);

	# underfit:
	plot_contour(X, y, alpha = 100.0, degree=6, ranges=None);

	plt.show();
