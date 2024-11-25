import numpy as np
from scipy.special import jv as Bv
from scipy.special import gamma, gammainc
import scipy as sp
import math


def gaussian_ff(x, Rg):
    # Gaussian chain form factor. From given x (q) and Rg the function returns the form factor of an ideal chain
    qrg = (x * Rg) ** 2
    fa = (2 * (np.exp(-qrg) - 1 + qrg)) / (qrg ** 2)
    return fa


def debye(qrg):
    return (1 - np.exp(-qrg ** 2)) / (qrg ** 2)


def stable_gaussian_ff(x, Rg):
    # Gaussian chain form factor with modification for small q
    qrg = (x * Rg)
    fa = np.where(qrg > 1e-5, debye(qrg) ** 2, np.exp(1 - qrg ** 2 / 3))
    return fa


def guinier_ff(x, Rg):
    return np.exp(-(x * Rg) ** 2 / 3)


def _fa_sphere(qr):
    # Scattering amplitude of a sphere, from given qr (q * R), where R is the sphere radius
    fa = np.ones_like(qr)

    qr1 = qr[qr > 0]
    fa[qr > 0] = 3 * (np.sin(qr1) - qr1 * np.cos(qr1)) / (qr1 ** 3)
    return fa


def sphere(x, Rg):
    # Form factor of a sphere, from given x(q) and sphere radius R.
    R = Rg * np.sqrt(5 / 3)
    return _fa_sphere(np.abs(x) * R) ** 2


def r_alpha(R, E, a):
    # Auxiliary function to be used for the ellipsoid form factor calculation
    return R * np.sqrt(np.sin(a) ** 2 + E ** 2 * np.cos(a) ** 2)


def integration(func, min_i, max_i, iterations):
    # Integration function. Inputs: func (the function which it integrates with),
    # min_i (the lower bounds of the integration), max_i (the upper bounds of the integration), iterations
    # (the number of iterations to be used to calculate the integration). Returns the result of the integration.
    res = 0
    integration_params = np.linspace(min_i, max_i, iterations)
    del_ip = (max_i - min_i) / iterations
    for ip in integration_params:
        res += del_ip * func(ip)
    return res


def ellipsoid(q, R, Rg):
    # Form factor of a three-dimensional ellipsoid, with radii (R, R, R + epsilon*R). Inputs are q, R, epsilon.
    int_it = 10
    epsilon = np.sqrt(5 * ((Rg / R) ** 2) - 2)

    res = np.zeros(len(q))
    for i in range(len(q)):
        res[i] = sp.integrate.quad(lambda a: np.sin(a) * _fa_sphere(q[i] * r_alpha(R, epsilon, a)) ** 2,
                                   0, np.pi / 2)[0]
    return res


def _fa_cylinder_(a, q, R, L):
    # Auxiliary function to be used for the cylinder form factor calculation. Inputs are a, q, R, L.
    return (2 * Bv(1, q * R * np.sin(a)) / (q * R * np.sin(a))) * (
            np.sin(q * L * np.cos(a / 2)) / (q * L * np.cos(a / 2)))


def cylinder(q, R, RG):
    # Form factor of a cylinder, with radius R and length L. Inputs are  q, R, L
    int_it = 10
    L = np.sqrt(6 * (RG ** 2 - 2 * R ** 2))
    res = np.zeros(len(q))
    for i in range(len(q)):
        res[i] = sp.integrate.quad(lambda a: np.sin(a) * _fa_cylinder_(a, q[i], R, L) ** 2, 0.001, np.pi / 2)[0]
    return res


def polysphere(q, Rg, variance):
    # Form factor of a polysphere, with radius Rg and variance. Inputs are q, Rg, variance.
    prg = sp.stats.norm(Rg, np.sqrt(variance))

    def spq(x, inte):
        return sphere(q[inte], np.sqrt(5 / 2 * x ** 2))

    res = np.zeros(len(q))
    for I in range(len(q)):
        res[I] = sp.integrate.quad(lambda x: prg.pdf(x) * spq(x, I), Rg - np.sqrt(variance) * 3,
                                   Rg + np.sqrt(variance) * 3)[0]
    return res


def hammouda_sgc(x, rg, nu):
    U = (1 / 6) * (x ** 2 * rg ** 2 * (2 * nu + 1) * (2 * nu + 2))
    nup = 1 / (2 * nu)
    Up = U ** nup
    gamma_U = gammainc(nup, U) * gamma(nup)
    gamma_U2 = gammainc(nup * 2, U) * gamma(nup * 2)
    I = (1. / (nu * Up)) * (gamma_U - (1 / Up) * gamma_U2)
    return I


def straight_line(x, rg):
    return np.where(x * rg < 1000, np.ones_like(x), np.ones_like(x) * 1e-16)


def polyP(x, form_factor, r_values, distribution):
    # Create a mesh grid for x and r values
    X, R = np.meshgrid(x, r_values, indexing='ij')
    P_values = form_factor(X, R)
    # Perform the integration as a sum over the r dimension
    integral = np.sum(P_values * distribution, axis=1) * np.diff(r_values)[0]
    return integral


def generate_gaussian_distribution(r_mean, r_std, num_points=1000):
    """
    Generate a Gaussian distribution of radii values and corresponding weights.

    This function calculates a Gaussian distribution of rg values centered at `r_mean` with a standard deviation of
    `r_std`.
    The distribution is generated using a linearly spaced array of `num_points` within three standard deviations
    from the mean. The weights of the distribution are calculated using the Gaussian function.

    Parameters:
    r_mean (float): The mean of the Gaussian distribution.
    r_std (float): The standard deviation of the Gaussian distribution.
    num_points (int, optional): The number of points to generate in the distribution. Default is 1000.

    Returns:
    tuple: A tuple containing two numpy arrays: `r_values` and `gaussian_weights`.
        - `r_values`: An array of rg values.
        - `gaussian_weights`: An array of corresponding weights for each rg value.
    """
    # Generate the rg value array
    r_values = np.linspace(np.max([r_mean - 3 * r_std, 0]), r_mean + 3 * r_std, num_points)
    # Generate the corresponding gaussian function
    gaussian_weights = gaussian_function(r_values, r_mean, r_std)
    return r_values, gaussian_weights


def gaussian_function(x, A, B):
    exponent = np.exp(-0.5 * ((x - A) / B) ** 2) * (1 / (B * np.sqrt(2 * np.pi)))
    return exponent
