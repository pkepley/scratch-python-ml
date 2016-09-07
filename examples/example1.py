import sys
sys.addpath('../')
from linear_regression import *

def example1(partId):
	X1 = np.zeros((20, 1));
	X1[:,0] = np.exp(1) + np.exp(2) * np.linspace(0.1, 2.0, 20);
	Y1 = X1 + np.sin( np.ones(X1.shape) ) + np.cos(X1);
	
	X2 = np.zeros((len(X1), 3));
	X2[:,0] = X1[:,0];
	X2[:,1] = X1[:,0]**0.5;
	X2[:,2] = X1[:,0]**0.25;
	Y2 = Y1 ** 0.5 + Y1;
	
	if partId == '1':
		print 'This was a trivial exercise. Not including'
	
	if partId == '2':
		lr = linear_regression(X1, Y1);
		print lr.J(np.array([0.5, -0.5]), 0.0)[0]
		
	elif partId == '3':
		lr = linear_regression(X1, Y1);
		lr.eta = 0.01
		lr.max_iterations = 10
		lr.w = np.array([0.5, -0.5])
		lr.fit(use_preset_w = True)
		print lr.w
		
	elif partId == '4':
		Z2 = X2[:,:];
		mus = np.mean(Z2, axis=0)
		sigmas = np.std(Z2, axis=0, ddof=1)
		Z2 = (X2 - mus) / sigmas
		print Z2

	elif partId == '5':
		lr = linear_regression(X2, Y2)
		print lr.J( np.array([0.1, 0.2, 0.3, 0.4]), alpha = 0)[0]

	elif partId == '6':
		lr = linear_regression(X2, Y2)
		lr.w = np.array([-0.1, -0.2, -0.3, -0.4]) 
		lr.eta = 0.01
		lr.max_iterations = 10
		lr.fit(use_preset_w = True)
		print lr.w

	elif partId == '7':
		#normalEqn(X2, Y2)
		lr = linear_regression(X2,Y2)
		lr.fit_normal_eq()
		print lr.w
