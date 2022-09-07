import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import GP_kernels


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


def plot_GP(X_test, mu, cov, X_train=None, Y_train=None, samples=None,
            draw_ci=False):
    '''Plot gaussian process'''
    X_test = X_test.flatten()
    mu = mu.flatten()
    std = np.sqrt(np.diag(cov))
    
    if draw_ci:
        for std_i in np.linspace(2*std,0,21):
            plt.fill_between(X_test, mu-std_i, mu+std_i,
                             color=cm.tab10(4), alpha=0.02)
    if samples is not None:
        plt.plot(X_test, samples, '-', lw=.5)
    plt.plot(X_test, mu, 'k')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'kx', mew=1.0)
    plt.xlim([X_test.min(), X_test.max()])
    plt.ylim([(mu-3*std).min(), (mu+3*std).max()])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$', rotation=0)


def GP_predictor(X_test, X_train, Y_train, kernel_fun, kernel_args, sigma_n=1e-8):
    '''
    Computes the suffiсient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_test.
    
    Args:
        X_test: New input locations (n x d)
        X_train: Training locations (m x d)
        Y_train: Training targets (m x 1)
        kernel_fun: Kernel length parameter
        kernel_args: Kernel vertical variation parameter
        sigma_n: Noise parameter
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n)
    '''
    
    K_11 = kernel_fun(X_train, X_train, kernel_args) \
         + sigma_n**2*np.eye(len(X_train))
    K_12 = kernel_fun(X_train, X_test, kernel_args)
    K_solved = np.linalg.solve(K_11, K_12).T
    K_22 = kernel_fun(X_test,  X_test, kernel_args)
    
    mu  = K_solved @ Y_train
    cov = K_22 - K_solved @ K_12
    
    return mu, cov
