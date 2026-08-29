import numpy as np
from scipy import stats
from scipy.special import voigt_profile

def gaussian(x, amplitude, mean, sigma):
    x = np.asarray(x, dtype=np.float64)
    return amplitude*np.exp(-0.5*((x-mean)/sigma)**2)

def double_gaussian(x, amplitude, mean, sigma_core, core_fraction, sigma_tail):
    core = gaussian(x, amplitude, mean, sigma_core)
    tail = gaussian(x, amplitude, mean, sigma_tail)
    return core_fraction*core + (1.0-core_fraction)*tail

def crystal_ball(x, amplitude, beta, m, mean, sigma):
    values = stats.crystalball.pdf(x, beta, m, loc=mean, scale=sigma)
    peak = stats.crystalball.pdf(mean, beta, m, loc=mean, scale=sigma)
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return amplitude*values/peak

def voight(x, amplitude, mean, sigma, gamma):
    x = np.asarray(x, dtype=np.float64)
    values = voigt_profile(x - mean, sigma, gamma)
    peak = voigt_profile(0.0, sigma, gamma)
    return amplitude*values/peak

def exponential_background(x, amplitude, slope, pivot):
    x = np.asarray(x, device=np.float64)
    exponent = np.clip(slope*(x-pivot), -700.0, 700.0)
    return amplitude*np.exp(exponent)

def linear_background(x, intercept, slope, pivot):
    x = np.asarray(x, dtype=np.float64)
    return intercept + slope*(x - pivot)

def quadratic_background(x, intercept, slope, curvature, pivot):
    shifted = np.asarray(x, dtype=np.float64) - pivot
    return intercept + slope*shifted + curvature*shifted**2