import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Plot bivariate distribution
def generate_gauss_surface(mean, covariance, n_mesh=101):
    '''Generate 2d gauss density surface'''
    x = y = np.linspace(-5, 5, num=n_mesh)
    xx, yy = np.meshgrid(x, y) # Generate grid
    pdf = np.zeros_like(xx)
    # Fill the cost matrix for each combination of weights
    pdf = stats.multivariate_normal.pdf(
        np.dstack((xx, yy)), mean, covariance)
    return xx, yy, pdf


def plot_gp(mu, cov, X_test, X_train=None, Y_train=None, samples=[]):
    '''Plot gaussian process'''
    X_test = X_test.ravel()
    mu = mu.ravel()
    std = np.sqrt(np.diag(cov))
    
    plt.figure(figsize=(8, 5))
    plt.fill_between(
        X_test, mu - 2*std, mu + 2*std,
        color='grey', alpha=0.1, label='$\pm 2\,\sigma$')
    if len(samples):
        plt.plot(X_test, samples, '-', lw=.5)
    plt.plot(X_test, mu, 'k', label='среднее значение')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'kx', mew=1.5)
    plt.xlim([X_test.min(), X_test.max()])
    plt.ylim([(mu-3*std).min(), (mu+3*std).max()])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper right')


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    '''Plot approximation'''
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    '''Plot acquisition'''
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()    

def plot_convergence(X_sample, Y_sample, n_init=2):
    '''Plot convergence'''
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)
    
    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')

