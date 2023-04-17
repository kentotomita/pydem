from numba import njit, float64
from math import erf, sqrt, pi

@njit(float64(float64, float64, float64), fastmath=True)
def cdf(x, mu, sigma):
    """
    Approximated analytical function for the CDF of a normal distribution.
    
    Parameters
    ----------
    x : float
        The value at which to evaluate the CDF.
    mu : float
        The mean of the normal distribution.
    sigma : float
        The standard deviation of the normal distribution.
    
    Returns
    -------
    float
        The value of the CDF at the given value.
    """
    z = (x - mu) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(z))


x = 0
p = cdf(x, 0, 1)
print(f"The CDF of a standard normal distribution at x={x:.4f} is approximately {p:.4f}")
