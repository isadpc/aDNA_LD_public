"""
  Estimating jump-rates for the LS Model in real ancient data
"""

import os
import sys
import numpy as np
import allel
import pandas as pd
from tqdm import tqdm

sys.path.append('src/')
from li_stephens import *

# NOTE: we can swap this out for the full dataset with lower-coverage individuals as well...
ancient_samples = pd.read_csv('data/hap_copying/chrX_male_analysis/sample_lists/ancient_5x_individuals.txt', sep='\t')


def gen_raw_haps(x_chrom_data, panel_file, test_id='NA20827.SG'):
    """
        Function to filter x-chromosomal panel and fill in missing data
        and will eliminate positions with non-segregating variation as well
    """
    tot_x_data = np.load(x_chrom_data, allow_pickle=True)
    sample_ids = tot_x_data['samples'].astype(str)
    cm_pos = tot_x_data['cm_pos']
    bp_pos = tot_x_data['bp_pos']
    gt_data = tot_x_data['gt']

    panel_indivs = np.loadtxt(panel_file, dtype=str)
    gt_panel = gt_data[:,np.isin(sample_ids, panel_indivs)].T
    # assert that we have no missing snps in the panel
    assert(np.all((gt_panel == 0) | (gt_panel == 1)))
    ns = np.sum((gt_panel < 0), axis=0)

    indiv_hap = gt_data[:,np.isin(sample_ids, np.array([test_id]))].T
    indiv_hap = indiv_hap[0,:]


    cm_pos_filt = cm_pos / 1e2
    # All of the entries should be either zero or 1
    assert(np.all(gt_panel >= 0) & np.all(gt_panel < 2))
    return(gt_panel, cm_pos, bp_pos, indiv_hap)

def gen_filt_panels(x_chrom_data, panel_file, test_id='NA20827.SG', fill_hwe=False):
    """
        Function to filter x-chromosomal panel and fill in missing data
        and will eliminate positions with non-segregating variation as well
    """
    tot_x_data = np.load(x_chrom_data, allow_pickle=True)
    sample_ids = tot_x_data['samples'].astype(str)
    cm_pos = tot_x_data['cm_pos']
    gt_data = tot_x_data['gt']


    panel_indivs = np.loadtxt(panel_file, dtype=str)
    gt_panel = gt_data[:,np.isin(sample_ids, panel_indivs)].T
    # assert that we have the
    assert(np.all((gt_panel == 0) | (gt_panel == 1)))
    ns = np.sum((gt_panel < 0), axis=0)

    freq_panel = np.sum((gt_panel > 0), axis=0) / np.sum((gt_panel >= 0), axis=0)
    if fill_hwe:
        for i in range(freq_panel.size):
            if freq_panel[i] <= 0.:
                geno = np.zeros(ns[i], dtype=np.int8)
            elif freq_panel[i] >= 1:
                geno = np.ones(ns[i], dtype=np.int8)
            else:
                geno = binomial(1, p=freq_panel[i], size=ns[i])
            gt_panel[(gt_panel[:,i] < 0), i] = geno

    indiv_hap = gt_data[:,np.isin(sample_ids, np.array([test_id]))].T
    indiv_hap = indiv_hap
    indiv_hap[indiv_hap == 2] = 1
    indiv_hap = indiv_hap[0,:]

    # Q : What happens if we don't filter the monomorphics?
    mono_morphic = (freq_panel <= 0.) | (freq_panel >= 1.0)
    missing_test_hap = (indiv_hap < 0)
    bad_idx = (mono_morphic | missing_test_hap)

    gt_panel_filt = gt_panel[:,~bad_idx]
    cm_pos_filt = cm_pos[~bad_idx] / 1e2
    indiv_hap_filt = indiv_hap[~bad_idx]
    # All of the entries should be either zero or 1
    assert(np.all(gt_panel_filt >= 0) & np.all(gt_panel_filt < 2))
    return(gt_panel_filt, cm_pos_filt, indiv_hap_filt)


rule estimate_jump_rate_sample_real_1kg:
  """
    For a given sample and individuals in a reference panel
      estimate both the jump rate and the error parameter
  """
  input:
    hap_panel_chrX_1kg = 'data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.{panel}.real_1kg.chrX.male_only.recmap_{rec}.total.npz',
    panel_indivs_file = 'data/hap_copying/chrX_male_analysis/panel_files/{panel}.real_1kg.male_only.indivs.txt'
  output:
    mle_hap_copying_res=config['tmpdir'] +  'hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)'
  run:
    # NOTE : we should keep the raw panel and raw query haplotype as well here ...
    raw_panel, raw_cmpos, raw_bppos, raw_testhap = gen_raw_haps(x_chrom_data=input.hap_panel_chrX_1kg, panel_file=input.panel_indivs_file, test_id=str(wildcards.sample))
    panel, pos, test_hap = gen_filt_panels(x_chrom_data=input.hap_panel_chrX_1kg, panel_file=input.panel_indivs_file, test_id=str(wildcards.sample), fill_hwe=True)
    ls_model = LiStephensHMM(haps=panel, positions=pos)
    n = 10
    jump_rates = np.logspace(2,5,n)
    log_ll_est = np.zeros(n, dtype=np.float32)
    for j in tqdm(range(n)):
      log_ll_est[j] = -ls_model.negative_logll(test_hap, scale=jump_rates[j])
    scale_inf_res = ls_model.infer_scale(test_hap, method='Bounded', bounds=(1.,1e6), tol=1e-4)
    # Setting the error rate to be similar to the original LS-Model
    mle_params = ls_model.infer_params(test_hap, x0=[1e2, 1e-3], bounds=[(1e1,1e7),(1e-6,0.9)], tol=1e-4)
    cur_params = np.array([np.nan,np.nan])
    se_params = np.array([np.nan,np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
      se_params = np.array([np.sqrt(mle_params.hess_inv.todense()[0,0]), np.sqrt(mle_params.hess_inv.todense()[1,1])])
    # getting some model stats
    model_stats = np.array([ls_model.nsnps, ls_model.nsamples])
    np.savez_compressed(output.mle_hap_copying_res, hap_panel=ls_model.haps, query_hap=test_hap, positions=ls_model.positions, raw_panel=raw_panel, raw_bppos=raw_bppos, raw_cmpos=raw_cmpos, raw_query_hap=raw_testhap, jump_rates=jump_rates, logll=log_ll_est, mle_params=cur_params, se_params=se_params, scale_inf=scale_inf_res['x'], model_stats=model_stats, sampleID=np.array([str(wildcards.sample)]))


rule gen_all_hap_copying_real1kg_panel:
  input:
     expand(config['tmpdir'] + 'hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz', rec='deCODE', panel=['ceu', 'eur', 'fullkg'], sample=ancient_samples['indivID'].values[0])


# TODO : final rule to full generate this dataset ...
# rule collapse_mle_hapcopying_results:
#   input:
#     expand('data/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{{panel}}.sample_{sample}.recmap_{rec}.listephens_hmm.npz', rec='deCODE', sample=ancient_samples['indivID'].values)
#   output:
#     'data/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.total.ls_stats.csv'
#   run:

# rule collapse_all:
