import numpy as np


# Isotropic squared exponential kernel
def gauss(X1, X2, kernel_args):
    '''
    Isotropic squared exponential kernel.
    Computes a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d)
        X2: Array of n points (n x d)
        l: Kernel length parameter
        sigma_f: Kernel vertical variation parameter

    Returns:
        Covariance matrix (m x n)
    '''
    l = kernel_args['l'] if 'l' in kernel_args.keys() else 1.
    sigma_f = kernel_args['sigma_f'] if 'sigma_f' in kernel_args.keys() else 1.
    sqdist = np.sum(X1**2,1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


# Brownian motion kernel
def brownian(X1, X2, kernel_args):
    '''
    Brownian motion kernel

    Args:
        X1: Array of m points (m x d)
        X2: Array of n points (n x d)
        sigma_f: Kernel vertical variation parameter

    Returns:
        Covariance matrix (m x n)
    '''
    sigma_f = kernel_args['sigma_f'] if 'sigma_f' in kernel_args.keys() else 1.
    cov = np.min(np.dstack(np.meshgrid(X1, X2)), axis=-1)
    return sigma_f**2 * cov.T


# Rational Quadratic Kernel
def rational_quadratic(X1, X2, kernel_args):
    '''
    Rational quadratic kernel

    Args:
        X1: Array of m points (m x d)
        X2: Array of n points (n x d)
        l: Kernel length parameter
        sigma_f: Kernel vertical variation parameter
        alpha: alpha parameter of th 
    
    Returns:
        Covariance matrix (m x n)
    '''
    l = kernel_args['l'] if 'l' in kernel_args.keys() else 1.
    sigma_f = kernel_args['sigma_f'] if 'sigma_f' in kernel_args.keys() else 1.
    alpha = kernel_args['alpha'] if 'alpha' in kernel_args.keys() else 1.
    sqdist = np.sum(X1**2,1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * (1 + sqdist / (2*alpha*l**2))**(-alpha)

# Periodic Kernel
def periodic(X1, X2, kernel_args):
    '''
    Periodic kernel
    
    Args:
        X1: Array of m points (m x d)
        X2: Array of n points (n x d)
        l: Kernel length parameter
        sigma_f: Kernel vertical variation parameter
        period: p parameter of th kernel
    
    Returns:
        Covariance matrix (m x n)
    '''
    l = kernel_args['l'] if 'l' in kernel_args.keys() else 1.
    sigma_f = kernel_args['sigma_f'] if 'sigma_f' in kernel_args.keys() else 1.
    period = kernel_args['period'] if 'period' in kernel_args.keys() else 1.
    dist = (np.sum(X1**2,1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T))**0.5
    return sigma_f**2 * np.exp(-2*np.sin(np.pi*dist/period)**2 / l**2)