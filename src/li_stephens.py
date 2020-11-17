"""Implementation of the Li-Stephens Copying HMM."""
import numpy as np
from numpy.random import binomial
from numba import jit
from scipy.optimize import minimize, minimize_scalar


@jit(nopython=True, cache=True)
def _log_sum_exp(arr):
    """Log-sum-exponential trick."""
    a = np.max(arr)
    x = arr - a
    tot = a + np.log(np.sum(np.exp(x)))
    return tot


@jit(nopython=True, cache=True)
def _emission_helper(a, hij, eps):
    """Numba function to compute the emission probability."""
    prob_mut = eps
    prob_no_mut = 1.0 - eps
    # Calculating emission probability here
    emiss_prob = (prob_mut + prob_no_mut) * (a == hij) + prob_mut * (a != hij)
    return np.log(emiss_prob)


@jit(nopython=True, cache=True)
def _forward_algo(haps, positions, test_hap, eps, nsamples, nsnps, scale):
    """Numba Function to compute the forward algorithm."""
    assert haps.shape[1] == positions.size
    assert test_hap.size == positions.size
    assert scale > 0
    alphas_null = [
        _emission_helper(test_hap[0], haps[j, 0], eps) for j in range(nsamples)
    ]
    alphas = np.zeros(shape=(nsamples, nsnps), dtype=np.float64)
    alphas[:, 0] = np.array(alphas_null, dtype=np.float64)
    start_pos = positions[0]
    # Iterating through the snps
    for i in range(1, nsnps):
        cur_pos = positions[i]
        # Calculate distance between samples
        dij = cur_pos - start_pos
        # Probability of recombination before coalescence
        rate = scale * dij
        # Probability of moving away (maybe use log1p here?)
        pj = 1.0 - np.exp(-rate)
        # This second term is the log-transitions
        second_term = alphas[:, i - 1] - np.log(nsamples) + np.log(pj)
        a = _log_sum_exp(second_term)
        # Calculating the underlying likelihood
        for j in range(nsamples):
            # Calculate probability of no transition away from the state
            no_trans = np.log(1.0 - pj) + alphas[j, i - 1]
            # Calculate updated alpha parameter
            x = np.array([a, no_trans])
            alphas[j, i] = _emission_helper(
                test_hap[i], haps[j, i], eps
            ) + _log_sum_exp(x)
        start_pos = cur_pos
    return alphas


class LiStephensHMM:
    """Simple class to apply the Li-Stephens HMM."""

    def __init__(self, haps, positions):
        """Initialize the haplotype copying model.

        Args:
            haps (`:np.array:`): numpy array of 0/1.

        """
        assert haps.shape[1] == positions.size
        self.haps = haps
        self.positions = positions
        self.nsamples = self.haps.shape[0]
        self.nsnps = self.positions.size
        self.eps = 0.001

    def negative_logll(self, test_hap, scale=1.0, eps=0.001):
        """Negative log-likelihood calculation for a test haplotype."""
        assert test_hap.size == self.nsnps
        alphas = _forward_algo(
            self.haps, self.positions, test_hap, eps, self.nsamples, self.nsnps, scale
        )
        #         alphas = self.forward(test_hap, scale, eps)
        neg_logll = -np.nansum(alphas[:, -1])
        return neg_logll

    def infer_scale(self, test_hap, eps=0.001, **kwargs):
        """Inferring scale by minimizing the marginal negative log-likelihood."""
        assert eps >= 0.0
        f = lambda s: self.negative_logll(test_hap, scale=s, eps=eps)  # noqa
        ta = minimize_scalar(f, **kwargs)
        return ta

    def infer_params(self, test_hap, x0=[1e2, 1e-2], **kwargs):
        """Infer joint parameters using numerical optimization."""
        f = lambda c: self.negative_logll(test_hap, scale=c[0], eps=c[1])  # noqa
        x = minimize(f, x0=x0, **kwargs)
        return x

    def sim_haplotype(self, scale=1e2, eps=1e-2, seed=None):
        """Simulate a haplotype from the LS model to test against."""
        if seed is not None:
            np.random.seed(seed)
        haps = self.haps
        rec_pos = self.positions
        nhaps, nsnps = haps.shape[0], haps.shape[1]
        sim_hap = np.zeros(nsnps)
        idxs = np.zeros(nsnps, dtype=np.uint32)
        # setup the initial random sample
        cur_pos = rec_pos[0]
        idxs[0] = np.random.randint(nhaps)
        sim_hap[0] = np.random.choice(
            [haps[idxs[0], 0], 1 - haps[idxs[0], 0]], p=[eps, 1.0 - eps]
        )
        for i in range(1, nsnps):
            # What is the probability that you jump?
            dij = rec_pos[i] - cur_pos
            pj = 1.0 - np.exp(-(scale * dij))
            if np.random.binomial(1, pj):
                idxs[i] = np.random.randint(nhaps)
            else:
                idxs[i] = idxs[i - 1]

            # set the current position
            cur_pos = rec_pos[i]
        for i in range(1, nsnps):
            sim_hap[i] = np.random.choice(
                [haps[idxs[i], i], 1 - haps[idxs[i], i]], p=[1.0 - eps, eps]
            )
        sim_hap = sim_hap.astype(np.int8)
        return (sim_hap, idxs)

    def fill_haplotypes_hwe(self):
        """Use HWE within the haplotype panel to fill in any missing variants.

        NOTE: missing variants are encoded as an nan within the panel.

        """
        assert np.sum(self.haps < 0) > 0
        # Estimate frequencies otherwise
        freqs = np.sum(self.haps, axis=0)
        ns = np.sum((self.haps < 0), axis=0)
        freqs = freqs / (self.n_samples - ns)
        assert freqs.size == ns.size
        assert freqs.size == self.n_snps
        haps_hwe_filled = np.copy(self.haps)
        for i in range(self.n_snps):
            if freqs[i] <= 0.0:
                geno = np.zeros(ns[i], dtype=np.int8)
            elif freqs[i] >= 1.0:
                geno = np.ones(ns[i], dtype=np.int8)
            else:
                geno = binomial(1, p=freqs[i], size=ns[i])
            haps_hwe_filled[self.haps[:, i] < 0, i] = geno
        # Fill in the haplotypes via the sampling procedure
        self.haps = haps_hwe_filled.astype(np.int8)

    def filter_sites(self, test_hap):
        """Filter to only sites on the haplotype panel that are typed in the test haplotype.

        NOTE: untyped variants are encoded as np.nan.

        """
        assert test_hap.size == self.n_snps
        idx = np.where(test_hap >= 0)[0]
        pos = np.copy(self.positions)
        haps = np.copy(self.haps)
        # Doing the indexing ...
        haps = haps[:, idx]
        pos = pos[idx]

        # resetting the system level variables
        self.haps = haps.astype(np.int8)
        self.positions = pos.astype(np.float64)
        self.n_snps = pos.size
