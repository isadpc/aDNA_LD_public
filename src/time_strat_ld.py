"""Classes to simulate two-locus tree statistics."""

import numpy as np
from tqdm import tqdm


class TimeStratifiedLDStats:
    """Class to compute LD statistics in time-stratified samples."""

    def joint_ld(
        haps, gen_pos, times, ta=100, maf=0.001, polymorphic_total=True, **kwargs
    ):
        """XXX."""
        assert haps.shape[0] == times.size
        assert haps.shape[1] == gen_pos.size
        assert ta >= 0
        assert haps.shape[0] % 2 == 0
        assert (maf >= 0) and (maf < 0.5)
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
        idx_x, idx_y = np.tril_indices(pos_filt.size, k=-1)
        return (eD0Dt_norm[idx_x, idx_y], dist[idx_x, idx_y], af_mod, af_anc)

    def joint_r2(haps, gen_pos, times):
        """Calculate r2 in the joint time-stratified setting."""
        pass
