"""
  Estimating jump-rates for the LS Model
"""

import os
import sys
import numpy as np
import msprime as msp
import tszip 
import allel
import pandas as pd
from tqdm import tqdm

sys.path.append('src/')
from li_stephens import *
from sim_copying_changepoints import obtain_topological_changes
from sim_copying_changepoints import count_changepoints
from aDNA_coal_sim import *

from scipy.optimize import minimize_scalar

# Defining relevant directories for data analysis
REICH_LAB_1240K_DATA_DIR = '/scratch/midway2/abiddanda/reich_lab_data/data/'
RECOMB_RATE_DIR= 'data/maps_b37/'

rec_map_types = {"Physical_Pos" : np.uint64, 
                 "deCODE" : np.float32, 
                 "COMBINED_LD" : np.float32, 
                 "YRI_LD": np.float32, 
                 "CEU_LD" : np.float32, 
                 "AA_Map": np.float32, 
                 "African_Enriched" : np.float32,
                 "Shared_Map" : np.float32}



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


rule filt_male_1240k_x_chrom:
  """
    Filter to only males on the x-chromosomes for VCF 
  """
  input:
    vcf_file = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.vcf.gz',
    ind_file = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.ind',
    chrom_rename = REICH_LAB_1240K_DATA_DIR + 'rename_x_chrom.txt'
  output:
    male_inds = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.males_only.ind',
    vcf_male_only = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.male_only.vcf.gz',
    vcf_male_only_idx = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.male_only.vcf.gz.tbi',
  shell:
    """
    awk \'$2 == \"M\" {{print $1}}\' {input.ind_file} > {output.male_inds}
    bcftools view -S {output.male_inds} --force-samples {input.vcf_file} | bcftools annotate --rename-chrs {input.chrom_rename} | bgzip -@4 > {output.vcf_male_only}
    tabix -f {output.vcf_male_only}
    """

    
rule interpolate_gen_hap_panel_X_chrom:
  """
     Linearly interpolate recombination position into the full X-chromosome panel
  """
  input:
    vcf = rules.filt_male_1240k_x_chrom.output.vcf_male_only,
    genmap = RECOMB_RATE_DIR + 'maps_chr.X'
  output:
    panel_file='data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.panel.v42.4.1240K.chrX.male_only.recmap_{rec}.total.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)' 
  run:
    # Reading in VCF
    vcf_data = allel.read_vcf(input.vcf)
    geno = vcf_data['calldata/GT']
    geno_summed = np.sum(geno, axis=2)
    # NOTE: summed up but we will try to filter to non-missing SNPs in the initial setting ... 
    geno_summed = geno_summed.astype(np.int8)
    # Generating the list of positions
    pos = vcf_data['variants/POS']
    # Reference Alleles
    ref_alleles = vcf_data['variants/REF']
    # Alternative Alleles
    alt_alleles = vcf_data['variants/ALT']
    sample_IDs = vcf_data['samples']
    # Reading in the recombination map 
    rec_df = pd.read_csv(input.genmap, sep='\s+', low_memory=True, dtype=rec_map_types)
    rec_pos = rec_df['Physical_Pos'].values
    rec_dist = rec_df[str(wildcards.rec)].values
    subsequent_distance = rec_dist[:-1] - rec_dist[1:]
    idx = np.where(subsequent_distance != 0)[0]
    min_pos, max_pos = rec_pos[np.min(idx)], rec_pos[np.max(idx)]
    # Filter to the minimul
    filt_idx = np.where((pos <= max_pos) & (pos >= min_pos))[0]
    real_pos_filt = pos[filt_idx]
    gt_filt = geno_summed[filt_idx,:]
    ref_alleles = ref_alleles[filt_idx]
    alt_alleles = alt_alleles[filt_idx]
    interp_rec_pos = np.interp(real_pos_filt, rec_pos[idx], rec_dist[idx])
    # Saving the file
    np.savez_compressed(output.panel_file, gt=gt_filt, samples=sample_IDs, ref=ref_alleles, alt=alt_alleles, bp_pos = real_pos_filt, cm_pos = interp_rec_pos)


rule estimate_jump_rate_sample:
  """
    For a given sample and individuals in a reference panel
      estimate both the jump rate and the error parameter
  """
  input:
    hap_panel = rules.interpolate_gen_hap_panel_X_chrom.output.panel_file,
    panel_indivs_file = 'data/hap_copying/chrX_male_analysis/panel_files/{panel}.indivs.txt'
  output:
    mle_hap_copying_res='data/hap_copying/chrX_male_analysis/mle_est/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)' 
  run:
    panel, pos, test_hap = gen_filt_panels(x_chrom_data=input.hap_panel, panel_file=input.panel_indivs_file, test_id=str(wildcards.sample), fill_hwe=True)
    ls_model = LiStephensHMM(haps=panel, positions=pos)
    ls_model.theta = ls_model._infer_theta()
    jump_rates = np.logspace(2,5,50)
    log_ll_est = np.zeros(50, dtype=np.float32)
    for j in tqdm(range(50)):
      log_ll_est[j] = -ls_model._negative_logll(test_hap, scale=jump_rates[j])
    scale_inf_res = ls_model._infer_scale(test_hap, method='Bounded', bounds=(1.,1e6), tol=1e-5)  
    # Setting the error rate to be similar to the original LS-Model
    mle_params = ls_model._infer_params(test_hap, x0=[1e2, 1e-3], bounds=[(1e1,1e7),(1e-6,0.1)], tol=1e-5)
    cur_params = np.array([np.nan,np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    # getting some model stats
    model_stats = np.array([ls_model.n_snps, ls_model.n_samples])
    np.savez_compressed(output.mle_hap_copying_res, hap_panel=ls_model.haps, query_hap=test_hap, positions=ls_model.positions, jump_rates=jump_rates, logll=log_ll_est, mle_params=cur_params, scale_inf=scale_inf_res['x'], model_stats=model_stats, sampleID=np.array([str(wildcards.sample)]))
  
  

# --------- Using the Real 1000 Genomes as a panel ----------- #
# REICH_LAB_1240K_DATA_DIR = '/scratch/midway2/abiddanda/reich_lab_data/data/'
DATA_DIR_1KG = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/haps/'

ancient_samples = pd.read_csv('data/hap_copying/chrX_male_analysis/sample_lists/ancient_individuals.txt', sep='\t')

rule generate_chrX_1kg_panel_vcf:
  input:
    chrX_vcf_1kg = DATA_DIR_1KG + 'ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz',
    panel_indivs = 'data/hap_copying/chrX_male_analysis/panel_files/{panel}.real_1kg.male_only.indivs.txt',
    reich_lab_chrX_1240K_male_only_vcf = rules.filt_male_1240k_x_chrom.output.vcf_male_only,
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)',
    panel = '(ceu|eur|fullkg)'
  output:
    tmp_chrX_pos = temp('data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.pos'),
    vcf = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.vcf.gz',
    vcf_idx = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.vcf.gz.tbi'
  shell:
    """
      bcftools query -f \'%CHROM\t%POS\n\' {input.reich_lab_chrX_1240K_male_only_vcf} > {output.tmp_chrX_pos}
      bcftools view -v snps -m2 -M2 -S {input.panel_indivs} -R {output.tmp_chrX_pos} {input.chrX_vcf_1kg} | bcftools sort | bgzip -@5 > {output.vcf}
      tabix -f {output.vcf}
    """


rule merge_real_1kg_vcf_chrX_1240K_panel:
  input:
    vcf_1240k_only = rules.filt_male_1240k_x_chrom.output.vcf_male_only,
    vcf_real_1kg =  rules.generate_chrX_1kg_panel_vcf.output.vcf,
  output:
    tmp_chrX_pos = temp('data/hap_copying/chrX_male_analysis/vcf_real1kg/merged/chrX.{panel}.real_1kg.v42_merged.1240K_only.male_only.sites.pos'),
    vcf = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/merged/chrX.{panel}.real_1kg.v42_merged.1240K_only.male_only.vcf.gz'
  shell:
    """
      bcftools query -f \"%CHROM\t%POS\n\" {input.vcf_real_1kg} > {output.tmp_chrX_pos}
      bcftools merge {input.vcf_1240k_only} {input.vcf_real_1kg} -R {output.tmp_chrX_pos} | bgzip -@4 > {output.vcf}
      tabix -f {output.vcf}
    """
    

rule interp_gen_hap_panel_chrX_real1kg:
  """
     Linearly interpolate recombination position into the full X-chromosome panel
     from the 1000 Genomes Project Phase 3 data 
  """
  input:
    vcf = rules.merge_real_1kg_vcf_chrX_1240K_panel.output.vcf,
    genmap = RECOMB_RATE_DIR + 'maps_chr.X'
  output:
    panel_file='data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.{panel}.real_1kg.chrX.male_only.recmap_{rec}.total.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)',
    panel = '(ceu|eur|fullkg)'
  run:
    # Reading in VCF
    vcf_data = allel.read_vcf(input.vcf)
    geno = vcf_data['calldata/GT']
    # Taking the first haplotype here since it should be equivalent ... 
    geno_summed = geno[:,:,0]
    geno_summed = geno_summed.astype(np.int8)
    geno_summed[geno_summed == 2] = 1
    # Generating the list of positions
    pos = vcf_data['variants/POS']
    # Reference Alleles
    ref_alleles = vcf_data['variants/REF']
    # Alternative Alleles
    alt_alleles = vcf_data['variants/ALT']
    sample_IDs = vcf_data['samples']
    # Reading in the recombination map 
    rec_df = pd.read_csv(input.genmap, sep='\s+', low_memory=True, dtype=rec_map_types)
    rec_pos = rec_df['Physical_Pos'].values
    rec_dist = rec_df[str(wildcards.rec)].values
    subsequent_distance = rec_dist[:-1] - rec_dist[1:]
    idx = np.where(subsequent_distance != 0)[0]
    min_pos, max_pos = rec_pos[np.min(idx)], rec_pos[np.max(idx)]
    # Filter to the minimul
    filt_idx = np.where((pos <= max_pos) & (pos >= min_pos))[0]
    real_pos_filt = pos[filt_idx]
    gt_filt = geno_summed[filt_idx,:]
    ref_alleles = ref_alleles[filt_idx]
    alt_alleles = alt_alleles[filt_idx]
    interp_rec_pos = np.interp(real_pos_filt, rec_pos[idx], rec_dist[idx])
    # Saving the file
    np.savez_compressed(output.panel_file, gt=gt_filt, samples=sample_IDs, ref=ref_alleles, alt=alt_alleles, bp_pos = real_pos_filt, cm_pos = interp_rec_pos)
    

rule estimate_jump_rate_sample_real_1kg:
  """
    For a given sample and individuals in a reference panel
      estimate both the jump rate and the error parameter
  """
  input:
    hap_panel_chrX_1kg = rules.interp_gen_hap_panel_chrX_real1kg.output.panel_file,
    panel_indivs_file = 'data/hap_copying/chrX_male_analysis/panel_files/{panel}.real_1kg.male_only.indivs.txt'
  output:
    mle_hap_copying_res='data/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)' 
  run:
    # NOTE : we should keep the raw panel and raw query haplotype as well here ...
    raw_panel, raw_cmpos, raw_bppos, raw_testhap = gen_raw_haps(x_chrom_data=input.hap_panel_chrX_1kg, panel_file=input.panel_indivs_file, test_id=str(wildcards.sample))
    panel, pos, test_hap = gen_filt_panels(x_chrom_data=input.hap_panel_chrX_1kg, panel_file=input.panel_indivs_file, test_id=str(wildcards.sample), fill_hwe=True)
    ls_model = LiStephensHMM(haps=panel, positions=pos)
    ls_model.theta = ls_model._infer_theta()
    n = 10
    jump_rates = np.logspace(2,5,n)
    log_ll_est = np.zeros(n, dtype=np.float32)
    for j in tqdm(range(n)):
      log_ll_est[j] = -ls_model._negative_logll(test_hap, scale=jump_rates[j])
    scale_inf_res = ls_model._infer_scale(test_hap, method='Bounded', bounds=(1.,1e6))  
    # Setting the error rate to be similar to the original LS-Model
    mle_params = ls_model._infer_params(test_hap, x0=[1e2, 1e-3], bounds=[(1e1,1e7),(1e-6,0.9)], tol=1e-7)
    cur_params = np.array([np.nan,np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    # getting some model stats
    model_stats = np.array([ls_model.n_snps, ls_model.n_samples])
    np.savez_compressed(output.mle_hap_copying_res, hap_panel=ls_model.haps, query_hap=test_hap, positions=ls_model.positions, raw_panel=raw_panel, raw_bppos=raw_bppos, raw_cmpos=raw_cmpos, raw_query_hap=raw_testhap, jump_rates=jump_rates, logll=log_ll_est, mle_params=cur_params, scale_inf=scale_inf_res['x'], model_stats=model_stats, sampleID=np.array([str(wildcards.sample)]))


rule gen_all_hap_copying_real1kg_panel:
  input:
     expand('data/hap_copying/chrX_male_analysis/mle_est_real_1kg/chrX_filt.panel_{panel}.sample_{sample}.recmap_{rec}.listephens_hmm.npz', rec='deCODE', panel=['ceu', 'eur'], sample=ancient_samples['indivID'].values) 


# TODO : final rule to full generate this dataset ... 
