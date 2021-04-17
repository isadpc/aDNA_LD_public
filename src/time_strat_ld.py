"""Classes to simulate two-locus tree statistics."""

import numpy as np
from tqdm import tqdm


class TimeStratifiedLDStats:
    """Class to compute LD statistics in time-stratified samples."""

    def joint_ld(
        haps,
        gen_pos,
        times,
        ta=100,
        maf=0.001,
        max_dist=0.01,
        polymorphic_total=True,
        **kwargs
    ):
        """Compute the joint LD statistics here."""
        assert haps.shape[0] == times.size
        assert haps.shape[1] == gen_pos.size
        assert ta >= 0
        assert haps.shape[0] % 2 == 0
        assert (maf >= 0) and (maf < 0.5)
        if ta == 0:
            x = int(haps.shape[0] / 2)
            mod_idx = np.arange(x)
            anc_idx = np.arange(x, haps.shape[0])
        else:
            mod_idx = np.where(times == 0)[0]
            anc_idx = np.where(times == ta)[0]
        gt_mod = haps[mod_idx, :]
        af_mod = np.mean(gt_mod, axis=0)
        gt_anc = haps[anc_idx, :]
        af_anc = np.mean(gt_anc, axis=0)
        gt_tot = np.vstack([gt_mod, gt_anc])
        af_tot = np.mean(gt_tot, axis=0)
        if polymorphic_total:
            af_filt_jt = (af_tot > maf) & (af_tot < 1.0 - maf)
        else:
            af_filt_jt = (
                (af_mod > maf)
                & (af_anc > maf)
                & (af_mod < 1.0 - maf)
                & (af_anc < 1.0 - maf)
            )
        # Perform some filtering using the MAF filters ...
        pos_filt = gen_pos[af_filt_jt]
        gt_mod = gt_mod[:, af_filt_jt]
        gt_anc = gt_anc[:, af_filt_jt]
        af_mod = af_mod[af_filt_jt]
        af_anc = af_anc[af_filt_jt]
        # Calculate the product of joint covariance D0Dt
        numerator = np.cov(gt_mod.T, **kwargs) * np.cov(gt_anc.T, **kwargs)
        dist = np.zeros(shape=(pos_filt.size, pos_filt.size), dtype=np.float32)
        denom = np.zeros(shape=(pos_filt.size, pos_filt.size), dtype=np.float32)
        for i in tqdm(range(pos_filt.size)):
            # The distance in recombination distances
            dist[:, i] = np.sqrt((pos_filt - pos_filt[i]) ** 2)
            denom[:, i] = af_mod[i] * (1.0 - af_anc[i]) * af_mod * (1.0 - af_anc)
        # Normalizing the product of LD
        eD0Dt_norm = numerator / denom
        # We should grab the maximal distance here ...
        #         idx = np.where(dist < max_dist)
        return (eD0Dt_norm[(dist < max_dist)], dist[(dist < max_dist)], af_mod, af_anc)

    def time_strat_hap_freq(
        haps,
        gen_pos,
        times,
        ta=100,
        maf=0.001,
        dist_bins=None,
        m=100,
        polymorphic_total=True,
        seed=42,
    ):
        """Calculate the haplotype frequencies within specific distance bins.

        Arguments:
        ----------
        * haps: (np.array) - a N x M array of haplotypes
        * gen_pos: (np.array) - length M vectors of genetic positions
        * times: (np.array) - length N vector of times for each haploid sequencing
        * ta: (int) - timepoint for ancient simulations
        * maf: (float) - minimum minor allele frequency here
        * dist_bins (np.array) - distance bins for estimating the haplotype frequencies
        * m (int) - number of monte-carlo samples per

        Returns:
        --------
        * xxx
        * xxx

        """
        assert haps.shape[0] == times.size
        assert haps.shape[1] == gen_pos.size
        assert ta >= 0
        assert haps.shape[0] % 2 == 0
        assert (maf >= 0) and (maf < 0.5)
        np.random.seed(seed)
        if ta == 0:
            x = int(haps.shape[0] / 2)
            mod_idx = np.arange(x)
            anc_idx = np.arange(x, haps.shape[0])
        else:
            mod_idx = np.where(times == 0)[0]
            anc_idx = np.where(times == ta)[0]
        gt_mod = haps[mod_idx, :]
        af_mod = np.mean(gt_mod, axis=0)
        gt_anc = haps[anc_idx, :]
        af_anc = np.mean(gt_anc, axis=0)
        gt_tot = np.vstack([gt_mod, gt_anc])
        af_tot = np.mean(gt_tot, axis=0)
        if polymorphic_total:
            af_filt_jt = (af_tot > maf) & (af_tot < 1.0 - maf)
        else:
            af_filt_jt = (
                (af_mod > maf)
                & (af_anc > maf)
                & (af_mod < 1.0 - maf)
                & (af_anc < 1.0 - maf)
            )
        # Perform some filtering using the MAF filters ...
        pos_filt = gen_pos[af_filt_jt]
        # We have to filter the genotype matrices as well...
        gt_mod = gt_mod[:, af_filt_jt]
        gt_anc = gt_anc[:, af_filt_jt]
        dist = np.zeros((pos_filt.size, pos_filt.size), dtype=np.float32)
        for i in range(pos_filt.size):
            dist[i, :] = np.sqrt((pos_filt[i] - pos_filt) ** 2)
        if dist_bins == "logspaced":
            # We want to conduct log-spaced binning  now.
            dist_bins = np.logspace(np.log10(dist.min()), np.log10(dist.max()), 50)
        elif dist_bins == "auto":
            _, dist_bins = np.histogram(dist, bins="auto")
        elif dist_bins is None:
            dist_bins = np.linspace(dist.min(), dist.max(), 50)
        assert dist_bins.size > 2
        pABmod = []
        pABanc = []
        pAmod = []
        pBmod = []
        pAanc = []
        pBanc = []
        Danc = []
        Dmod = []
        gen_dist = []
        for i in tqdm(range(1, dist_bins.size)):
            # Get indices in this range
            idx1, idx2 = np.where((dist > dist_bins[i - 1]) & (dist < dist_bins[i]))
            # sample K haplotype pairs via Monte-Carlo
            for j in np.random.choice(idx1.size, size=m):
                # Choose the same two snps ...
                mod_haps = gt_mod[:, [idx1[j], idx2[j]]]
                anc_haps = gt_anc[:, [idx1[j], idx2[j]]]

                # Calculate the relevant haplotype frequencies ...
                pAB_mod = (
                    np.sum((mod_haps[:, 0] == 1) & (mod_haps[:, 1] == 1)) / mod_idx.size
                )
                pAB_anc = (
                    np.sum((anc_haps[:, 0] == 1) & (anc_haps[:, 1] == 1)) / anc_idx.size
                )
                pA_mod = np.sum(mod_haps[:, 0]) / mod_idx.size
                pB_mod = np.sum(mod_haps[:, 1]) / mod_idx.size
                pA_anc = np.sum(anc_haps[:, 0]) / anc_idx.size
                pB_anc = np.sum(anc_haps[:, 1]) / anc_idx.size
                assert pAB_mod <= np.min([pA_mod, pB_mod])
                assert pAB_anc <= np.min([pA_anc, pB_anc])
                # Calculate the ancient and modern LD statistics ...
                D_mod = pAB_mod - pA_mod * pB_mod
                D_anc = pAB_anc - pA_anc * pB_anc
                pABmod.append(pAB_mod)
                pABanc.append(pAB_anc)
                pAmod.append(pA_mod)
                pBmod.append(pB_mod)
                pAanc.append(pA_anc)
                pBanc.append(pB_anc)
                Dmod.append(D_mod)
                Danc.append(D_anc)
                gen_dist.append(dist[idx1[j], idx2[j]])
        # Convert all to numpy arrays
        pABmod = np.array(pABmod)
        pABanc = np.array(pABanc)
        pAmod = np.array(pAmod)
        pBmod = np.array(pBmod)
        pAanc = np.array(pAanc)
        pBanc = np.array(pBanc)
        Dmod = np.array(Dmod)
        Danc = np.array(Danc)
        gen_dist = np.array(gen_dist)
        # Make sure all the positions are positive numbers
        assert np.all(gen_dist >= 0)
        # Return all of the statistics as numpy arrays
        return (pABmod, pABanc, pAmod, pAanc, pBmod, pBanc, Dmod, Danc, gen_dist)

    def time_strat_two_locus_haps(
        haps, gen_pos, times, ta=0.1, maf=0.001, polymorphic_total=True
    ):
        """Calculate the haplotype frequencies across two loci.

        Arguments:
        ----------
        * haps: (np.array) - a N x M array of haplotypes
        * gen_pos: (np.array) - length M vectors of genetic positions
        * times: (np.array) - length N vector of times for each haploid sequencing
        * ta: (int) - timepoint for ancient simulations
        * maf: (float) - minimum minor allele frequency here
        * polymorphic_total: (bool)

        Returns:
        --------
        pABmod, pAmod, pBmod

        """
        assert haps.shape[0] == times.size
        assert haps.shape[1] == gen_pos.size
        assert ta >= 0
        assert haps.shape[0] % 2 == 0
        assert (maf >= 0) and (maf < 0.5)
        if ta == 0:
            x = int(haps.shape[0] / 2)
            mod_idx = np.arange(x)
            anc_idx = np.arange(x, haps.shape[0])
        else:
            mod_idx = np.where(times == 0)[0]
            anc_idx = np.where(times == ta)[0]
        gt_mod = haps[mod_idx, :]
        af_mod = np.mean(gt_mod, axis=0)
        gt_anc = haps[anc_idx, :]
        af_anc = np.mean(gt_anc, axis=0)
        gt_tot = np.vstack([gt_mod, gt_anc])
        af_tot = np.mean(gt_tot, axis=0)
        if polymorphic_total:
            af_filt_jt = (af_tot > maf) & (af_tot < 1.0 - maf)
        else:
            af_filt_jt = (
                (af_mod > maf)
                & (af_anc > maf)
                & (af_mod < 1.0 - maf)
                & (af_anc < 1.0 - maf)
            )
        # Perform some filtering using the MAF filters ...
        pos_filt = gen_pos[af_filt_jt]
        dist = np.zeros((pos_filt.size, pos_filt.size), dtype=np.float32)
        for i in range(pos_filt.size):
            dist[i, :] = np.sqrt((pos_filt[i] - pos_filt) ** 2)
        # We have to filter the genotype matrices as well...
        gt_mod = gt_mod[:, af_filt_jt]
        gt_anc = gt_anc[:, af_filt_jt]
        # Get indices that are across loci
        idx1, idx2 = np.where((dist > 1) & (dist <= 2))
        pABmod = np.zeros(idx1.size)
        pABanc = np.zeros(idx1.size)
        pAmod = np.zeros(idx1.size)
        pBmod = np.zeros(idx1.size)
        pAanc = np.zeros(idx1.size)
        pBanc = np.zeros(idx1.size)
        Danc = np.zeros(idx1.size)
        Dmod = np.zeros(idx1.size)
        gen_dist = np.zeros(idx1.size)
        # sample K haplotype pairs via Monte-Carlo
        for j in range(idx1.size):
            # Choose the same two snps ...
            mod_haps = gt_mod[:, [idx1[j], idx2[j]]]
            anc_haps = gt_anc[:, [idx1[j], idx2[j]]]

            # Calculate the relevant haplotype frequencies ...
            pAB_mod = (
                np.sum((mod_haps[:, 0] == 1) & (mod_haps[:, 1] == 1)) / mod_idx.size
            )
            pAB_anc = (
                np.sum((anc_haps[:, 0] == 1) & (anc_haps[:, 1] == 1)) / anc_idx.size
            )
            pA_mod = np.sum(mod_haps[:, 0]) / mod_idx.size
            pB_mod = np.sum(mod_haps[:, 1]) / mod_idx.size
            pA_anc = np.sum(anc_haps[:, 0]) / anc_idx.size
            pB_anc = np.sum(anc_haps[:, 1]) / anc_idx.size
            assert pAB_mod <= np.min([pA_mod, pB_mod])
            assert pAB_anc <= np.min([pA_anc, pB_anc])
            # Calculate the ancient and modern LD statistics ...
            D_mod = pAB_mod - pA_mod * pB_mod
            D_anc = pAB_anc - pA_anc * pB_anc
            pABmod[j] = pAB_mod
            pABanc[j] = pAB_anc
            pAmod[j] = pA_mod
            pBmod[j] = pB_mod
            pAanc[j] = pA_anc
            pBanc[j] = pB_anc
            Dmod[j] = D_mod
            Danc[j] = D_anc
            gen_dist[j] = dist[idx1[j], idx2[j]]
        # Make sure all the positions are positive numbers
        assert np.all(gen_dist >= 0)
        # Return all of the statistics as numpy arrays
        return (pABmod, pABanc, pAmod, pAanc, pBmod, pBanc, Dmod, Danc, gen_dist)
