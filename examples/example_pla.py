import sys
sys.path.append('../')
from pla import *

import matplotlib.pyplot as plt

def generate_lin_sep_data(n):
    # generate 2d data:
    d = 2
    
    # generate random numbers in -1,1
    X = np.zeros(n*d)
    X = np.random.uniform(-1,1,n*d)
    X = X.reshape((n,d))
    
    # generate two points to generate line:
    xp = np.random.uniform(-1,1,2*d)
    xp = xp.reshape((2,d))
    # line direction vector:
    v = xp[1,:] - xp[0,:]
    # normal to line:
    w = np.array([v[1],-v[0]])
    
    # generate classifications:
    # points on the "positive" side get labeled 1
    # points on the "negative" side get labeled -1                         
    y = np.array(map(lambda p : 2*int(np.dot(w, p - xp[0,:]) >= 0)-1, X))
    return X, y, xp

def demonstrate_fit(n_observations):
    # Estimate the number of iterations:
    X, y, xp = generate_lin_sep_data(n_observations)
    model = pla(X,y)
    model.fit()
    
    # points lying on the PLA Decision Boundary:
    t = np.linspace(-1,1,10)
    s = -(model.w[0] + model.w[1] * t)/model.w[2]

    # points lying on the true class boundary:
    xp1 = xp[0,:]
    xp2 = xp[1,:]
    v = xp2 - xp1
    multiplier = 2.0 / np.linalg.norm(xp2 - xp1)
    xpp = np.zeros((2,2))
    xpp[0,:] = xp1 + multiplier * v
    xpp[1,:] = xp1 - multiplier * v

    # plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X[:,0], X[:,1], c=y, s=40, alpha=0.5)
    ax.plot(xpp[:,0],xpp[:,1], label='True Class Boundary')
    ax.plot(t,s, label='PLA Decision Boundary')
    plt.axis([-1,1,-1,1])
    ax.legend()

if __name__ == '__main__':
    n_runs = 1000
    n_observations = 100

    print 'Running Problem 1.4 from Learning from Data'
    print 'Using n_runs = {0}, n_observations = {1}'.format(n_runs, n_observations)

    n_demonstration_fits = 3
    print 'Demonstrating fit for %d example data sets' % n_demonstration_fits
    for i in range(n_demonstration_fits):
        demonstrate_fit(n_observations)
    plt.show()

    # Estimate the number of iterations:
    n_train_its = np.zeros(n_runs)
    for i in range(n_runs):
        X, y, xp = generate_lin_sep_data(n_observations)
        model = pla(X,y)
        model.fit()
        n_train_its[i] = model.n_iterations
    print 'Average number of iterations required to converge: %f' % np.mean(n_train_its)

    # Estimate the generalization error:
    prob_incorrect = np.zeros(n_runs)
    for i in range(n_runs):
        X, y, xp = generate_lin_sep_data(2 * n_observations)
        X_train, X_test = X[0:n_observations,:], X[n_observations+1:,:]
        y_train, y_test = y[0:n_observations], y[n_observations+1:]
                
        model = pla(X_train, y_train)
        model.fit()
        prob_incorrect[i] = len(np.where(y_test != model.predict(X_test))[0]) / float(n_observations)
    print 'Estimated P[f(x) == g(x)]: %f' % np.mean(prob_incorrect)
