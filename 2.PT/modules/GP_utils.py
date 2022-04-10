import numpy as np
from scipy import stats
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


def plot_GP(X_test, mu, cov, X_train=[], Y_train=[], samples=[],
            draw_ci=False):
    '''Plot gaussian process'''
    X_test = X_test.flatten()
    mu = mu.flatten()
    std = np.sqrt(np.diag(cov))
    
    if draw_ci:
        for std_i in np.linspace(2*std,0,11):
            plt.fill_between(X_test, mu-std_i, mu+std_i,
                             color='grey', alpha=0.02)
    if len(samples):
        plt.plot(X_test, samples, '-', lw=.5)
    plt.plot(X_test, mu, 'k')
    if len(X_train):
        plt.plot(X_train, Y_train, 'kx', mew=1.0)
    plt.xlim([X_test.min(), X_test.max()])
    plt.ylim([(mu-3*std).min(), (mu+3*std).max()])
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$', rotation=0)


def GP_predictor(X_test, X_train, Y_train, kernel_fun, kernel_args, sigma_y=1e-8):
    '''
    Computes the suffiсient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_test.
    
    Args:
        X_test: New input locations (n x d)
        X_train: Training locations (m x d)
        Y_train: Training targets (m x 1)
        kernel_fun: Kernel length parameter
        kernel_args: Kernel vertical variation parameter
        sigma_y: Noise parameter
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n)
    '''
    
    K_11 = kernel_fun(X_train, X_train, kernel_args) \
         + sigma_y*np.eye(len(X_train))
    K_12 = kernel_fun(X_train, X_test, kernel_args)
    K_solved = np.linalg.solve(K_11, K_12).T
    K_22 = kernel_fun(X_test,  X_test, kernel_args)
    
    mu  = K_solved @ Y_train
    cov = K_22 - K_solved @ K_12
    
    return mu, cov


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    '''Plot approximation'''
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.flatten(), mu.flatten()+2*std, mu.flatten()-2*std, alpha=0.1) 
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
    # plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].flatten()
    y = Y_sample[n_init:].flatten()
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

