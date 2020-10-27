"""Calculating expected time to the first coalescent for an ancient sample.

This is used to generate results for Appendix 3 of our paper.

"""

import numpy as np
from scipy.stats import norm


def griffiths_approx(k, t):
    """Approximation of expected number of lineages from Griffiths 1981."""
    eT = k / (k + (1 - k) * np.exp(-t / 2))
    return eT


def prob_j(nt, j):
    """Probability that the lineage joins at the jth coalescent event."""
    nt = np.int32(nt)
    p = 2.0 / j * np.prod([(1.0 - 2.0 / k) for k in range(j + 1, nt + 1)])
    return p


def time_to_first_coal(nt):
    """Time of first coalescent with the ancient sample."""
    # NOTE : here we are assuming we are just adding a single lineage in
    nt = np.int32(nt)
    i_s = np.arange(nt + 1, 2, -1)
    cum_sum_is = np.cumsum(2.0 / (i_s * (i_s - 1)))
    probs = np.array([prob_j(nt, np.int32(j)) for j in range(nt + 1, 2, -1)])
    return np.sum(probs * cum_sum_is)


def time_first_coal_griffiths(n, t, f=np.round):
    """Time to first coal event with the ancient lineage.

    Uses Griffiths approximation.

    Args:
    n - number of lineages.
    t - time of ancient sampling.
    f - function to map expectation to an integer value.

    """
    e_nt = np.int32(griffiths_approx(n, t))
    eTi = time_to_first_coal(f(e_nt))
    return eTi


def appx_num_lineages_mean_var(n, t):
    """Approximate the number of lineages from Chen & Chen 2013."""
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
    """Weight the lineages by the normal density."""
    # mean and variance of the number of lineages left
    mu, var = appx_num_lineages_mean_var(n, t)
    # integer value of the mean
    mu_int = np.int32(mu)
    idx = np.arange(max(1, mu_int - eps), min(mu_int + eps, n), dtype=np.int32)
    # weight using a discretized approximation to the normal density
    weights = norm.pdf(idx, loc=mu, scale=np.sqrt(var))
    weights = weights / np.sum(weights)
    return (idx, weights)


def full_approx_time_to_first_coal(n, t, eps=10):
    """Compute the fully weighted number of lineages as a function of time."""
    # The weightings of the number of lineages left
    nts, ws = weight_lineages(n, t)
    # The expected time to coalescence for each possibility
    eTi = []
    for n in nts:
        eTi.append(time_to_first_coal(n))
    eTi = np.array(eTi)
    eTi_total = np.sum(ws * eTi)
    return (eTi, nts, ws, eTi_total)
