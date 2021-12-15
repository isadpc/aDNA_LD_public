"""
  Estimating jump-rates for the LS Model in real ancient data
"""

import os
import sys
import numpy as np
import allel
import pandas as pd
import traceback
from tqdm import tqdm
from numpy.random import binomial
from scipy.interpolate import UnivariateSpline

sys.path.append("src/")
from li_stephens import LiStephensHMM


# Import configurations to add some things
configfile: "config.yml"


# NOTE: we can swap this out for the full dataset with lower-coverage individuals as well...
ancient_samples = pd.read_csv(
    "data/hap_copying/chrX_male_analysis/sample_lists/ancient_europe_individuals.txt",
    sep="\t",
)


def gen_raw_haps(x_chrom_data, panel_file, test_id="NA20827.SG"):
    """
    Function to filter x-chromosomal panel and fill in missing data
    and will eliminate positions with non-segregating variation as well
    """
    tot_x_data = np.load(x_chrom_data, allow_pickle=True)
    sample_ids = tot_x_data["samples"].astype(str)
    cm_pos = tot_x_data["cm_pos"]
    bp_pos = tot_x_data["bp_pos"]
    gt_data = tot_x_data["gt"]

    panel_indivs = np.loadtxt(panel_file, dtype=str)
    gt_panel = gt_data[:, np.isin(sample_ids, panel_indivs)].T
    # assert that we have no missing snps in the panel
    assert np.all((gt_panel == 0) | (gt_panel == 1))
    ns = np.sum((gt_panel < 0), axis=0)

    indiv_hap = gt_data[:, np.isin(sample_ids, np.array([test_id]))].T
    indiv_hap = indiv_hap[0, :]

    cm_pos_filt = cm_pos / 1e2
    # All of the entries should be either zero or 1
    assert np.all(gt_panel >= 0) & np.all(gt_panel < 2)
    return (gt_panel, cm_pos, bp_pos, indiv_hap)


def gen_filt_panels(x_chrom_data, panel_file, test_id="NA20827.SG", fill_hwe=False):
    """
    Function to filter x-chromosomal panel and fill in missing data
    and will eliminate positions with non-segregating variation as well
    """
    tot_x_data = np.load(x_chrom_data, allow_pickle=True)
    sample_ids = tot_x_data["samples"].astype(str)
    cm_pos = tot_x_data["cm_pos"]
    gt_data = tot_x_data["gt"]

    panel_indivs = np.loadtxt(panel_file, dtype=str)
    gt_panel = gt_data[:, np.isin(sample_ids, panel_indivs)].T
    # assert that we have the
    assert np.all((gt_panel == 0) | (gt_panel == 1))
    ns = np.sum((gt_panel < 0), axis=0)

    freq_panel = np.sum((gt_panel > 0), axis=0) / np.sum((gt_panel >= 0), axis=0)
    if fill_hwe:
        for i in range(freq_panel.size):
            if freq_panel[i] <= 0.0:
                geno = np.zeros(ns[i], dtype=np.int8)
            elif freq_panel[i] >= 1:
                geno = np.ones(ns[i], dtype=np.int8)
            else:
                geno = binomial(1, p=freq_panel[i], size=ns[i])
            gt_panel[(gt_panel[:, i] < 0), i] = geno

    indiv_hap = gt_data[:, np.isin(sample_ids, np.array([test_id]))].T
    indiv_hap = indiv_hap
    indiv_hap[indiv_hap == 2] = 1
    indiv_hap = indiv_hap[0, :]

    # Q : What happens if we don't filter the monomorphics?
    mono_morphic = (freq_panel <= 0.0) | (freq_panel >= 1.0)
    missing_test_hap = indiv_hap < 0
    bad_idx = mono_morphic | missing_test_hap

    gt_panel_filt = gt_panel[:, ~bad_idx]
    cm_pos_filt = cm_pos[~bad_idx] / 1e2
    indiv_hap_filt = indiv_hap[~bad_idx]
    # All of the entries should be either zero or 1
    assert np.all(gt_panel_filt >= 0) & np.all(gt_panel_filt < 2)
    return (gt_panel_filt, cm_pos_filt, indiv_hap_filt)


def calc_se_spline(scales, loglls, mle_scale):
    """Calculate the standard errors using the spline."""
    logll_spl = UnivariateSpline(scales, loglls, s=0, k=4)
    logll_deriv2 = logll_spl.derivative(n=2)
    se = 1.0 / np.sqrt(-logll_deriv2(mle_scale))
    return se


def calc_se_finite_diff(f, mle_x, eps=1e-1):
    """f is the loglikelihood function."""
    xs = np.array([mle_x - 2 * eps, mle_x - eps, mle_x, mle_x + eps, mle_x + 2 * eps])
    ys = np.array(
        [
            f(mle_x - 2 * eps),
            f(mle_x - eps),
            f(mle_x),
            f(mle_x + eps),
            f(mle_x + 2 * eps),
        ]
    )
    dy = np.diff(ys, 1)
    dx = np.diff(xs, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (xs[:-1] + xs[1:])
    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)
    ysecond = dyfirst / dxfirst
    se = 1.0 / np.sqrt(-ysecond[1])
    return (ysecond, se)


# TODO : what should be done
rule estimate_jump_rate_sample_real_1kg:
    """
    For a given sample and individuals in a reference panel
      estimate both the jump rate and the error parameter
    """
    input:
        hap_panel_chrX_1kg="data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.{panel}.real_1kg.chrX.male_only.recmap_{rec}.total.npz",
        panel_indivs_file="data/hap_copying/chrX_male_analysis/panel_files/{panel}.real_1kg.male_only.indivs.txt",
    output:
        mle_hap_copying_res=config["tmpdir"]
        + "hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz",
    wildcard_constraints:
        rec="(CEU_LD|deCODE)",
    run:
        n = 30
        scales = np.logspace(2, 6, n)
        neg_log_lls = np.zeros(n, dtype=np.float32)
        try:
            # NOTE : we should keep the raw panel and raw query haplotype as well here ...
            raw_panel, raw_cmpos, raw_bppos, raw_testhap = gen_raw_haps(
                x_chrom_data=input.hap_panel_chrX_1kg,
                panel_file=input.panel_indivs_file,
                test_id=str(wildcards.sample),
            )
            panel, pos, test_hap = gen_filt_panels(
                x_chrom_data=input.hap_panel_chrX_1kg,
                panel_file=input.panel_indivs_file,
                test_id=str(wildcards.sample),
                fill_hwe=True,
            )
            ls_model = LiStephensHMM(haps=panel, positions=pos)
            ls_model.theta = ls_model._infer_theta()
            print(ls_model.n_snps, ls_model.n_samples, ls_model.theta)
            for j in tqdm(range(n)):
                neg_log_lls[j] = ls_model._negative_logll(
                    test_hap, eps=1e-2, scale=scales[j]
                )
            min_idx = np.argmin(neg_log_lls)
            print(scales, neg_log_lls)
            scales_bracket = (1.0, scales[min_idx] + 1.0)
            scale_inf_res = ls_model._infer_scale(
                test_hap, method="Brent", bracket=scales_bracket
            )
            # Setting the error rate to be similar to the original LS-Model
            f = lambda x: -ls_model._negative_logll(test_hap, scale=x, eps=1e-2)
            _, se_finite_diff = calc_se_finite_diff(f, scale_inf_res.x)
            # NOTE: We initialize the x0 using the marginal solution for the scale?
            bounds = [(10, 1e4), (1e-4, 0.25)]
            mle_params = ls_model._infer_params(
                test_hap, x0=[scale_inf_res.x, 1e-2], bounds=bounds
            )
            cur_params = np.array([np.nan, np.nan])
            se_params = np.array([np.nan, np.nan])
            if mle_params["success"]:
                cur_params = mle_params["x"]
                se_params = np.array(
                    [
                        np.sqrt(mle_params.hess_inv.todense()[0, 0]),
                        np.sqrt(mle_params.hess_inv.todense()[1, 1]),
                    ]
                )
            # getting some model stats out
            model_stats = np.array([ls_model.n_snps, ls_model.n_samples])
            hap_panel = ls_model.haps
            pos = ls_model.positions
        except Exception:
            traceback.print_exc()
            cur_params = np.array([np.nan, np.nan])
            se_params = np.array([np.nan, np.nan])
            scale_inf_res = {"x": np.nan}
            se_finite_diff = np.nan
            model_stats = np.array([0, 0])
            hap_panel = np.nan
            test_hap = np.nan
            pos = np.nan
            raw_panel = np.nan
            raw_bppos = np.nan
            raw_cmpos = np.nan
            raw_testhap = np.nan
        np.savez_compressed(
            output.mle_hap_copying_res,
            hap_panel=hap_panel,
            query_hap=test_hap,
            positions=pos,
            raw_panel=raw_panel,
            raw_bppos=raw_bppos,
            raw_cmpos=raw_cmpos,
            raw_query_hap=raw_testhap,
            jump_rates=scales,
            logll=-neg_log_lls,
            mle_params=cur_params,
            se_params=se_params,
            scale_inf=scale_inf_res["x"],
            se_scale_finite_diff=se_finite_diff,
            model_stats=model_stats,
            panel=str(wildcards.panel),
            sampleID=str(wildcards.sample),
        )


rule gen_all_hap_copying_real1kg_panel:
    input:
        expand(
            config["tmpdir"]
            + "hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz",
            rec="deCODE",
            panel=["ceu", "eur", "fullkg"],
            sample=ancient_samples["indivID"].values,
        ),


rule collapse_mle_hapcopying_results:
    input:
        files=rules.gen_all_hap_copying_real1kg_panel.input,
    output:
        "results/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.total.ls_stats.csv",
    run:
        tot_rows_df = []
        for f in tqdm(input.files):
            # Load in the current data frame
            cur_df = np.load(f)
            # Keeping the fields that we want to keep around
            scales = cur_df["jump_rates"]
            logll = cur_df["logll"]
            scale_marginal = cur_df["scale_inf"]
            se_scale_finite_diff = cur_df["se_scale_finite_diff"]
            mle_params_jt = cur_df["mle_params"]
            se_params_jt = cur_df["se_params"]
            model_stats = cur_df["model_stats"]
            panel_ID = cur_df["panel"]
            sample_ID = cur_df["sampleID"]
            # calculate the se in the marginal scale using the spline operation as well...
            se_spline = calc_se_spline(scales, logll, mle_scale=scale_marginal)
            cur_row = [
                sample_ID,
                panel_ID,
                scale_marginal,
                se_spline,
                se_scale_finite_diff,
                mle_params_jt[0],
                se_params_jt[0],
                mle_params_jt[1],
                se_params_jt[1],
                model_stats[0],
                model_stats[1],
            ]
            tot_rows_df.append(cur_row)
        # Creating a final dataframe here ...
        final_df = pd.DataFrame(
            tot_rows_df,
            columns=[
                "indivID",
                "panelID",
                "scale_marginal",
                "se_marginal",
                "se_scale_marginal_fd",
                "scale_jt",
                "se_scale_jt",
                "eps_jt",
                "se_eps_jt",
                "nsnps",
                "nref_haps",
            ],
        )
        # join the data frame
        final_df.to_csv(str(output), index=False)


rule collapse_and_merge_data:
    """This rule combines the metadata file with the underlying parameter estimates."""
    input:
        file=rules.collapse_mle_hapcopying_results.output,
    output:
        "results/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.total.ls_stats.merged.csv",
    run:
        final_df = pd.read_csv(str(input.file))
        merged_df = pd.merge(ancient_samples, final_df, on="indivID", how="left")
        merged_df.to_csv(str(output), index=False)
