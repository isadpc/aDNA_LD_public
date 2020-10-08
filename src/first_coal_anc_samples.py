#!python3

"""
  Library of functions to calculate expected time to the first coalescent
"""

import numpy as np
from scipy.stats import norm

griffiths_approx = lambda k, t: k / (k + (1 - k) * np.exp(-t / 2))

# NOTE : the second version
def prob_j(nt, j):
    nt = np.int32(nt)
    p = 2.0 / j * np.prod([(1.0 - 2.0 / k) for k in range(j + 1, nt + 1)])
    return p


def prob_j2(n, l):
    """ Saunders 1984 - Corrolary 4"""
    i = n
    j = 2
    first_term = (j - 1) * (i + 1)
    second_term = fact(l) * fact(j) / fact(l + j)
    third_term = fact(i - l - 1) * fact(i - j) / (fact(i - 1) * fact(i - l - j + 1))
    total = first_term * second_term * third_term
    return total


def time_to_first_coal(nt):
    """
        Computes the expectation of the first coalescent time that involves
        the ancient sample.
    """
    # NOTE : here we are assuming we are just adding a single lineage in
    nt = np.int32(nt)
    i_s = np.arange(nt + 1, 2, -1)
    cum_sum_is = np.cumsum(2.0 / (i_s * (i_s - 1)))
    probs = np.array([prob_j(nt, np.int32(j)) for j in range(nt + 1, 2, -1)])
    return np.sum(probs * cum_sum_is)


def time_first_coal_griffiths(n, t, f=np.round):
    """
    Compute expected time to a first coalescent event involving the
    ancient lineage

    Args:
    n - number of lineages
    t - time of ancient sampling
    f - function to map expectation to an integer value
  """
    e_nt = np.int32(griffiths_approx(n, t))
    eTi = time_to_first_coal(f(e_nt))
    return eTi


def appx_num_lineages_mean_var(n, t):
    """
        Approximate the mean and variance of the number of lineages
        using results from Chen & Chen 2013
        n is the number of modern lineages you start with
    """
    # Defining the context here...
    beta = -0.5 * t
    alpha = 0.5 * n * t
    eta = (alpha * beta) / (alpha * (np.exp(beta) - 1.0) + beta * np.exp(beta))
    # Denote mean and variance
    mu = (2.0 * eta) / t
    # computing the variance here
    sigma2 = 2 * eta / t
    sigma2 = sigma2 * (eta + beta) ** 2
    sigma2 = sigma2 * (
        1 + eta / (eta + beta) - eta / alpha - eta / (alpha + beta) - 2 * eta
    )
    sigma2 = sigma2 / (beta ** 2)
    return (mu, sigma2)


def weight_lineages(n, t, eps=10):
    """
        Weight the lineages
    """
    # compute mean and variance of the number of lineages left
    mu, var = appx_num_lineages_mean_var(n, t)
    # compute the integer value of the mean
    mu_int = np.int32(mu)
    idx = np.arange(max(1, mu_int - eps), min(mu_int + eps, n), dtype=np.int32)
    # compute the weights using a discretized approximation to the normal density
    weights = norm.pdf(idx, loc=mu, scale=np.sqrt(var))
    weights = weights / np.sum(weights)
    return (idx, weights)


def full_approx_time_to_first_coal(n, t, eps=10):
    # Compute the weightings of the number of lineages left
    nts, ws = weight_lineages(n, t)
    # compute the expected time to coalescence for each possibility
    eTi = []
    for n in nts:
        eTi.append(time_to_first_coal(n))
    eTi = np.array(eTi)
    eTi_total = np.sum(ws * eTi)
    return (eTi, nts, ws, eTi_total)
