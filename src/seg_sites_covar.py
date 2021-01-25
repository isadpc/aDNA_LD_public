"""Methods to compute the correlation in segregating sites between samples."""

import numpy as np
import numpy.ma as ma
import pyranges
from pyranges import PyRanges
from scipy import interpolate

from scipy.stats import wilcoxon
from scipy.stats import binom


def extract_chrom_gen_pos(df, chrom=2):
    """Extract the genetic postition vector for a particular chromosome."""
    genpos = df[df["CHROM"] == chrom]["GENPOS"].values
    return genpos


def extract_phys_pos(df, chrom=2):
    """Get physical distance here."""
    physpos = df[df["CHROM"] == chrom]["POS"].values
    return physpos


def gen_binned_estimates(rec_dists, s1, s2, **kwargs):
    """Get binned estimates of the correlation in segregating sites."""
    _, bins = np.histogram(rec_dists, **kwargs)
    bin_idx = np.digitize(rec_dists, bins)
    bin_idx = bin_idx - 1

    # Setting up the accumulator vectors here
    rec_rate_mean = np.zeros(np.max(bin_idx))
    rec_rate_se = np.zeros(np.max(bin_idx))
    corr_s1_s2 = np.zeros(np.max(bin_idx))
    se_r = np.zeros(np.max(bin_idx))
    for i in range(np.max(bin_idx)):
        cur_idx = bin_idx == i
        n_pairs = np.sum(cur_idx)
        cur_rec_rates = rec_dists[cur_idx]
        rec_rate = np.nanmean(cur_rec_rates)
        se_rec_rate = np.nanstd(cur_rec_rates)

        # TODO : take concatenation of the masks
        corr_s1_s2_cur = ma.corrcoef(
            ma.masked_invalid(s1[cur_idx]), ma.masked_invalid(s2[cur_idx])
        )[0, 1]
        se_r_cur = np.sqrt((1.0 - corr_s1_s2_cur ** 2) / (n_pairs - 2))

        # Set the accumulators to return
        rec_rate_mean[i] = rec_rate
        rec_rate_se[i] = se_rec_rate
        corr_s1_s2[i] = corr_s1_s2_cur
        se_r[i] = se_r_cur

    # Return the different accumulators as we have them
    return (rec_rate_mean, rec_rate_se, corr_s1_s2, se_r)


def sign_test_corrSASB(corr_seg_obj_1, corr_seg_obj_2, test="binom", **kwargs):
    """Compute a non-parametric sign test."""
    assert corr_seg_obj_1.monte_carlo_results is not None
    assert corr_seg_obj_2.monte_carlo_results is not None
    monte_carlo_corr_res1 = corr_seg_obj_1.monte_carlo_results
    monte_carlo_corr_res2 = corr_seg_obj_2.monte_carlo_results
    # Make sure that they the have the same dimension
    x = monte_carlo_corr_res1[2, :]
    y = monte_carlo_corr_res2[2, :]
    diff = x - y
    assert monte_carlo_corr_res1.shape == monte_carlo_corr_res2.shape
    if test == "wilcoxon":
        stat, pval = wilcoxon(x, y, **kwargs)
    elif test == "binom":
        npos = np.sum(diff > 0)
        nneg = np.sum(diff < 0)
        n = x.size
        stat = np.mean(diff)
        pval = binom.cdf(np.min([nneg, npos]), n=n, p=0.5) + (
            1.0 - binom.cdf(np.max([npos, nneg]), n=n, p=0.5)
        )
    else:
        raise ValueError("Test is not supported!")
    return (diff, stat, pval)


class CorrSegSites:
    """Class to implement methods for the correlation in segregating sites."""

    def __init__(self):
        """Initialize the correlation in segregating sites."""
        # Add in a physical map distance for defining explicit windows
        self.phys_map = None
        # Three different dictionaries for
        self.recmap = None
        self.chrom_pos_dict = None
        self.chrom_physpos_dict = None
        self.chrom_weight_dict = None
        self.chrom_total_dict = None
        # Method to get the monte-carlo estimates to work
        self.rec_dist = None
        self.s1 = None
        self.s2 = None
        # Setting the Monte-Carlo Results here as none

    def _conv_cM_to_morgans(self):
        """Transform from cM to Morgans in case we need to."""
        assert self.chrom_pos_dict is not None
        for c in self.chrom_pos_dict:
            self.chrom_pos_dict[c] = self.chrom_pos_dict[c] / 100

    def _setids(self, ids=["X", "Y"]):
        """Set the ids for the individuals in question."""
        self.ids = ids

    def calc_windowed_seg_sites(self, chrom=0, L=1e3, filt_rec=True, mask=None):
        """Calculate windowed estimates of segregating sites.

        Arguments:
            * chrom: identifier for the chromosome
            * L: length of independent locus
            * filt_rec: filter recombination
            * mask: bed file for the underlying mask

        """
        assert self.chrom_pos_dict is not None
        phys_pos = self.chrom_physpos_dict[chrom]
        rec_pos = self.chrom_pos_dict[chrom]
        weights = self.chrom_weight_dict[chrom]
        if filt_rec:
            diff = np.abs(rec_pos[:-1] - rec_pos[1:])
            idx = np.where(diff != 0)[0]
            phys_pos = phys_pos[idx]
            rec_pos = rec_pos[idx]
            weights = weights[idx]
        if mask is not None:
            phys_pos = phys_pos.astype(np.float64)
            df_mask = pyranges.read_bed(mask)
            df_pos = PyRanges(chromosomes=chrom, starts=phys_pos, ends=(phys_pos + 1))
            cov_sites = df_pos.coverage(df_mask)
            sites_idx = np.array(cov_sites.FractionOverlaps.astype(np.float32))
            idx = np.where(sites_idx > 0.0)[0]
            phys_pos[idx] = np.nan
        # 1. Setup the bins for the analysis
        bins = np.arange(np.nanmin(phys_pos), np.nanmax(phys_pos), L)
        windowed_vars, bin_edges = np.histogram(
            phys_pos[~np.isnan(phys_pos)],
            bins=bins,
            weights=weights[~np.isnan(phys_pos)],
        )
        bin_edges = bin_edges.astype(np.uint32)
        # Interpolate the midpoints of the recombination bins
        f = interpolate.interp1d(phys_pos, rec_pos)
        midpts = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
        rec_midpts = f(midpts)
        # Calculate the weightings from the mask as needed ...
        mask_weights = np.ones(rec_midpts.size)
        if mask is not None:
            # Mask must be a bedfile
            df_windows = PyRanges(
                chromosomes=chrom, starts=bin_edges[:-1], ends=bin_edges[1:]
            )
            df_mask = pyranges.read_bed(mask)
            cov = df_windows.coverage(df_mask)
            mask_weights = np.array(cov.FractionOverlaps.astype(np.float32))
            # Set the mask weights to scale up the fraction that may be missing!
            mask_weights = 1.0 / (1.0 - mask_weights)
            mask_weights[np.isinf(mask_weights)] = np.nan

        # Stacking all of the data to make sure that we can use it later on
        tot_data = np.vstack([windowed_vars, bin_edges[1:], rec_midpts, mask_weights])
        self.chrom_total_dict[chrom] = tot_data

    def autocorr_sA_sB(self, sep=1):
        """Compute the autocorrelation across windows separated by a distance.

        NOTE : this is meant to be a faster
            alternative to the monte-carlo sampling approach

        """
        assert self.chrom_total_dict is not None
        corrs = []
        rec_dists = []
        for c in self.chrom_total_dict:
            # Grabbing the current version of the data for this chromosome
            cur_tot_data = self.chrom_total_dict[c]
            win_vars = cur_tot_data[0, :]
            rec_midpts = cur_tot_data[2, :]
            mask_weights = cur_tot_data[3, :]
            # Weight according to a mask - if it exists
            x = mask_weights * win_vars
            # Compute the empirical correlation here
            x1s = x[sep:]
            x2s = x[:-sep]
            # Setting the mask here
            a = ma.masked_invalid(x1s)
            b = ma.masked_invalid(x2s)
            corr_est = ma.corrcoef(a, b)[0, 1]
            # Should it be the mean or something else here
            rec_dist_mean = np.nanmean(rec_midpts[sep:] - rec_midpts[:-sep])
            corrs.append(corr_est)
            rec_dists.append(rec_dist_mean)
        corrs = np.array(corrs, dtype=np.float32)
        rec_dists = np.array(rec_dists, dtype=np.float32)
        return (rec_dists, corrs)

    def monte_carlo_corr_SA_SB(self, L=100, nreps=1000, chrom=22, seed=42):
        """Estimate the correlation in segregating sites using monte-carlo sampling."""
        assert self.chrom_physpos_dict is not None
        assert self.chrom_pos_dict is not None
        assert self.chrom_total_dict is not None

        if self.rec_dist is None:
            self.rec_dist = {}
            self.s1 = {}
            self.s2 = {}
        cur_tot_data = self.chrom_total_dict[chrom]
        windowed_het = cur_tot_data[0, :]
        rec_midpts = cur_tot_data[2, :]
        mask_weights = cur_tot_data[3, :]
        windowed_het_weighted = mask_weights * windowed_het
        rec_dists = []
        s1s = []
        s2s = []
        n = windowed_het.size
        for j in range(1, L):
            # Choose nreps indices
            idx = np.random.choice(np.arange(0, n - j), nreps)
            # TODO : note we do not mask here (or weight appropriately...)
            cur_s1s = windowed_het_weighted[idx]
            cur_s2s = windowed_het_weighted[idx + j]
            cur_rec_dists = rec_midpts[idx + j] - rec_midpts[idx]
            # Appending them to the
            rec_dists.append(cur_rec_dists)
            s1s.append(cur_s1s)
            s2s.append(cur_s2s)
        # vstacking and changing up the data types
        tot_rec_dists = np.hstack(rec_dists)
        s1s = np.hstack(s1s)
        s2s = np.hstack(s2s)
        self.rec_dist[chrom] = tot_rec_dists
        self.s1[chrom] = s1s
        self.s2[chrom] = s2s

    def gen_binned_rec_rate(self, chroms=None, **kwargs):
        """Get binned estimates of the correlation in segregating sites.

        Arguments:
            chroms - numpy array of

        """
        assert self.rec_dist is not None
        if chroms is None:
            rec_dists = np.hstack([self.rec_dist[i] for i in self.rec_dist])
            s1 = np.hstack([self.s1[i] for i in self.rec_dist])
            s2 = np.hstack([self.s2[i] for i in self.rec_dist])
        else:
            rec_dists = np.hstack([self.rec_dist[i] for i in chroms])
            s1 = np.hstack([self.s1[i] for i in chroms])
            s2 = np.hstack([self.s2[i] for i in chroms])

        # Filter out where either is a nan
        idx = ~np.isnan(rec_dists)
        rec_dists = rec_dists[idx]
        s1 = s1[idx]
        s2 = s2[idx]

        # check for nans in s1 and s2 as well
        s1_nans = ~np.isnan(s1)
        s2_nans = ~np.isnan(s2)
        idx_s1_s2 = s1_nans & s2_nans
        s1 = s1[idx_s1_s2]
        s2 = s2[idx_s1_s2]
        rec_dists = rec_dists[idx_s1_s2]

        # Running the method from outside of the class
        rec_rate_mean, rec_rate_se, corr_s1_s2, se_r = gen_binned_estimates(
            rec_dists, s1, s2, **kwargs
        )

        # Stack all the results to help and store for future use
        monte_carlo_results = np.vstack([rec_rate_mean, rec_rate_se, corr_s1_s2, se_r])
        self.monte_carlo_results = monte_carlo_results
        return (rec_rate_mean, rec_rate_se, corr_s1_s2, se_r)

    def monte_carlo_corr_SA_SB_v2(
        self, L=1e3, dist=100, nreps=1000, chrom=0, seed=42, filt_rec=True, mask=None
    ):
        """Estimate the correlation using alternative Monte-Carlo Sampling.

        Key: this allows us to test much shorter length scales

        """
        assert self.chrom_physpos_dict is not None
        assert self.chrom_pos_dict is not None
        assert L > 0
        assert dist > 0
        assert seed > 0
        np.random.seed(seed)
        phys_pos = self.chrom_physpos_dict[chrom]
        rec_pos = self.chrom_pos_dict[chrom]
        weights = self.chrom_weight_dict[chrom]
        if filt_rec:
            diff = np.abs(rec_pos[:-1] - rec_pos[1:])
            idx = np.where(diff != 0)[0]
            phys_pos = phys_pos[idx]
            rec_pos = rec_pos[idx]
            weights = weights[idx]
        if mask is not None:
            phys_pos = phys_pos.astype(np.float64)
            df_mask = pyranges.read_bed(mask)
            df_pos = PyRanges(chromosomes=chrom, starts=phys_pos, ends=(phys_pos + 1))
            cov_sites = df_pos.coverage(df_mask)
            sites_idx = np.array(cov_sites.FractionOverlaps.astype(np.float32))
            idx = np.where(sites_idx > 0.0)[0]
            phys_pos[idx] = np.nan
        # 1. Setup bins separated by some distance
        startp = np.nanmin(phys_pos)
        endp = startp + L
        windowed_vars = []
        bins = []
        while endp < np.nanmax(phys_pos):
            bins.append((startp, endp))
            start = np.searchsorted(phys_pos[~np.isnan(phys_pos)], startp, "left")
            end = np.searchsorted(phys_pos[~np.isnan(phys_pos)], endp, "right")
            # Append this to actually weight the variants
            windowed_vars.append(end - start)
            startp += L + dist
            endp += L + dist
        windowed_vars = np.array(windowed_vars)
        bin_edges = np.array(bins).ravel()
        #         print(bin_edges.size)
        #         print(windowed_vars.size, bin_edges.size)
        assert (bin_edges.size / 2) == windowed_vars.size
        # Interpolate the midpoints of the recombination bins
        f = interpolate.interp1d(phys_pos, rec_pos)
        rec_dist = f(bin_edges[2:-1:2]) - f(bin_edges[1:-2:2])
        #         print(np.mean(rec_dist))
        windowed_vars = windowed_vars[:-1]
        #         print(rec_dist.size, windowed_vars.size)

        # Calculate the weightings from the mask as needed ...
        mask_weights = np.ones(windowed_vars.size)
        if mask is not None:
            # Mask must be a bedfile
            df_windows = PyRanges(
                chromosomes=chrom, starts=bin_edges[:-1], ends=bin_edges[1:]
            )
            df_mask = pyranges.read_bed(mask)
            cov = df_windows.coverage(df_mask)
            mask_weights = np.array(cov.FractionOverlaps.astype(np.float32))
            # Set the mask weights to scale up the fraction that may be missing!
            mask_weights = 1.0 / (1.0 - mask_weights)
            mask_weights[np.isinf(mask_weights)] = np.nan

        # add to whatever total datatype that we require?
        windowed_het_weighted = mask_weights * windowed_vars
        #         print(windowed_het_weighted.size)
        s1s = windowed_het_weighted[:-2:2]
        s2s = windowed_het_weighted[1:-1:2]
        #         print(rec_dist.size, s1s.size, s2s.size)
        assert s1s.size == s2s.size
        #         assert ((rec_dist.size  / 2) - 1) == s1s.size
        # Perform the Monte-Carlo resampling here
        idx = np.random.randint(s1s.size, size=nreps)
        s1s_samp = s1s[idx]
        s2s_samp = s2s[idx]
        rec_dist_samp = rec_dist[2 * idx]
        if self.rec_dist is None:
            self.rec_dist = {}
            self.s1 = {}
            self.s2 = {}
        if chrom in self.rec_dist:
            tmp_rec_dist = np.append(self.rec_dist[chrom], rec_dist_samp)
            tmp_s1 = np.append(self.s1[chrom], s1s_samp)
            tmp_s2 = np.append(self.s2[chrom], s2s_samp)
            self.rec_dist[chrom] = tmp_rec_dist
            self.s1[chrom] = tmp_s1
            self.s2[chrom] = tmp_s2
        else:
            self.rec_dist[chrom] = rec_dist_samp
            self.s1[chrom] = s1s_samp
            self.s2[chrom] = s2s_samp


class CorrSegSitesSims(CorrSegSites):
    """Sub-class for computing the correlation in segregating sites."""

    def __init__(self):
        """Initialize the object."""
        super().__init__()

    def _load_data(self, hap_panel_file, mod_id=None):
        """Load in a haplotype panel which is a 2D binary vector."""
        if self.chrom_pos_dict is None:
            self.chrom_pos_dict = {}
            self.chrom_physpos_dict = {}
            self.chrom_weight_dict = {}
            self.chrom_total_dict = {}
            new_id = 0
        else:
            chrom_id = np.max([c for c in self.chrom_pos_dict])
            new_id = chrom_id + 1
        # Reading in the dataset
        df = np.load(hap_panel_file)
        phys_pos = df["phys_pos"]
        rec_pos = df["rec_pos"]
        hap_panel_test = df["haps"]

        # assuming the ancient haplotype is last
        anc_hap = hap_panel_test[-1, :]
        mod_haps = hap_panel_test[:-1, :]
        # Getting a modern haplotype
        n_mod = mod_haps.shape[0]
        mod_id = mod_id if (mod_id is not None) else np.random.randint(n_mod)
        mod_hap = mod_haps[mod_id, :]
        # Looking for segregating sites between the ancient and modern individuals
        idx = np.where(anc_hap != mod_hap)[0]
        # NOTE:  here we have setup a couple of
        #     rec_pos = rec_pos[idx].astype(np.float16)
        phys_pos = phys_pos[idx].astype(np.uint32)
        self.chrom_pos_dict[new_id] = rec_pos[idx]
        self.chrom_physpos_dict[new_id] = phys_pos[idx]
        self.chrom_weight_dict[new_id] = np.ones(np.sum(idx), dtype=np.float32)

    def _set_windows_missing(self, fmiss=0.1):
        """Set a fraction of windows to be randomly missing when computing correlations.

        NOTE: this method is really only good for testing

        """
        assert (fmiss < 1.0) & (fmiss > 0.0)
        assert self.chrom_total_dict is not None

        for c in self.chrom_total_dict:
            cur_tot_data = self.chrom_total_dict[c]
            n_win = cur_tot_data.shape[1]
            # Sample and set some windows to be nan
            idx = np.random.randint(0, high=n_win, size=int(fmiss * n_win))
            cur_tot_data[0, idx] = np.nan
            # reset the value within the class
            self.chrom_total_dict[c] = cur_tot_data

    def _set_window_masking(self, fmask=0.1, mean_mask=0.5, sigma_mask=0.1):
        """Set masking weights in scheme to be higher or lower.

        Arguments:
            fmask - fraction of windows to be masked
            mean_mask - average mask quality
            sigma_mask -

        """
        assert (fmask > 0.0) & (fmask <= 1.0)
        assert self.chrom_total_dict is not None

        for c in self.chrom_total_dict:
            cur_tot_data = self.chrom_total_dict[c]
            n_win = cur_tot_data.shape[1]
            # index of masked windows
            idx = np.random.randint(0, high=n_win, size=int(fmask * n_win))

            n_masked = idx.size
            mask_wts = np.random.normal(loc=mean_mask, scale=sigma_mask, size=n_masked)
            mask_wts[mask_wts < 0.0] = 0.0
            mask_wts[mask_wts >= 1.0] = 1.0
            # Rescale the number of variants in a window ...
            n_var_window = cur_tot_data[0, :]
            # We do not want to round for the testing here
            vars_reweighted = (1.0 - mask_wts) * n_var_window[idx]
            n_var_window[idx] = vars_reweighted
            cur_tot_data[0, :] = n_var_window
            # Now setting the masking weights
            cur_tot_data[3, idx] = 1.0 / (1.0 - mask_wts)
            cur_tot_data[3, np.isinf(cur_tot_data[3, :])] = np.nan

            # setting the values
            self.chrom_total_dict[c] = cur_tot_data

    def _set_sites_psuedohaploid(self, p_haploid=0.5):
        """Set some fraction of sites to be psuedo haploid."""
        assert (p_haploid >= 0.0) & (p_haploid < 1.0)
        assert self.chrom_weight_dict is not None
        for c in self.chrom_weight_dict:
            cur_weights = self.chrom_weight_dict[c]
            n_pos = cur_weights.size
            # index of masked windows
            idx = np.random.randint(0, high=n_pos, size=int(p_haploid * n_pos))
            cur_weights[idx] = 0.5
            self.chrom_weight_dict[c] = cur_weights


class CorrSegSitesRealDataHaploid(CorrSegSites):
    """Class for real data that is haploid."""

    def __init__(self):
        """Initialize the object."""
        super().__init__()

    def _load_data(self, filepath, miss_to_nan=False):
        """Load data from a single entry with a haploid modern and an ancient sample."""
        if self.chrom_pos_dict is None:
            self.chrom_pos_dict = {}
            self.chrom_physpos_dict = {}
            self.chrom_weight_dict = {}
            self.chrom_total_dict = {}
        df = np.load(filepath, allow_pickle="True")
        chrom = df["chrom"]
        phys_pos = df["pos"]
        gt_anc = df["gt_anc"]
        gt_mod_hap = df["gt_mod_hap"]
        rec_pos = df["rec_pos"]
        # Create the weights
        assert gt_anc.size == gt_mod_hap.size
        weights = np.zeros(gt_anc.size, dtype=np.float32)

        # Defining the different types of mismatches
        # Heterozygotes
        mis_match1 = (gt_mod_hap == 0) & (gt_anc == 1)
        mis_match2 = (gt_mod_hap == 1) & (gt_anc == 1)
        # Homozygotes here ...
        mis_match3 = (gt_mod_hap == 0) & (gt_anc == 2)
        mis_match4 = (gt_mod_hap == 1) & (gt_anc == 0)
        # Defining the relative weight given to each type of mismatch
        weights[mis_match1] = 0.5
        weights[mis_match2] = 0.5
        weights[mis_match3] = 1.0
        weights[mis_match4] = 1.0
        # Setting the element in this dictionary as this underlying function
        c = int(chrom[0])
        if miss_to_nan:
            idx = np.where((gt_anc + gt_mod_hap) <= 0)
            phys_pos = phys_pos.astype(np.float32)
            phys_pos[idx] = np.nan

        self.chrom_pos_dict[c] = rec_pos
        self.chrom_physpos_dict[c] = phys_pos
        self.chrom_weight_dict[c] = weights


class CorrSegSitesRealDataDiploid(CorrSegSites):
    """Class for real data where we have to compare two diploids."""

    def __init__(self):
        """Initialize the object."""
        super().__init__()

    def _load_data(self, filepath, miss_to_nan=False, weighting=False):
        """Load data from a single entry with a haploid modern and an ancient sample."""
        if self.chrom_pos_dict is None:
            self.chrom_pos_dict = {}
            self.chrom_physpos_dict = {}
            self.chrom_weight_dict = {}
            self.chrom_total_dict = {}

        df = np.load(filepath, allow_pickle="True")
        chrom = df["chrom"]
        phys_pos = df["pos"]
        gt_anc = df["gt_anc"]
        gt_mod = df["gt_mod"]
        rec_pos = df["rec_pos"]
        # Create the weights
        assert gt_anc.size == gt_mod.size
        weights = np.ones(gt_anc.size, dtype=np.float32)
        if weighting:
            # Defining the different types of mismatches
            mis_match1 = (gt_mod == 0) & (gt_anc == 1)
            mis_match2 = (gt_mod == 0) & (gt_anc == 1)
            mis_match3 = (gt_mod == 0) & (gt_anc == 2)
            mis_match4 = (gt_mod == 1) & (gt_anc == 0)
            # Defining the relative weight given to each type of mismatch
            weights[mis_match1] = 0.5
            weights[mis_match2] = 0.5
            weights[mis_match3] = 1.0
            weights[mis_match4] = 1.0
        # Setting the element in this dictionary as this underlying function
        c = int(chrom[0])
        if miss_to_nan:
            idx = np.where((gt_anc + gt_mod) <= 0)
            phys_pos = phys_pos.astype(np.float32)
            phys_pos[idx] = np.nan

        # Setting the dictionaries here ...
        self.chrom_pos_dict[c] = rec_pos
        self.chrom_physpos_dict[c] = phys_pos
        self.chrom_weight_dict[c] = weights
