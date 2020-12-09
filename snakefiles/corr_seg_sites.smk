#!python3

import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import allel

sys.path.append('src/')
from seg_sites_covar import *
from fit_corr_segsites import *

# Import configurations
configfile: "config.yml"

# Some simulations rely on the sims for haplotype copying
include: "hap_copying.smk"

# --- 1. Estimation from Simulations --- #
rule estimate_autocorr_sA_Sb_sims:
  input:
    expand(config['tmpdir'] + 'hap_copying/hap_panels/{{scenario}}/hap_panel_{{mod_n}}_{{n_anc}}_{{ta}}_{{length}}_Ne_{{Ne}}_{rep}.npz', rep=np.arange(20))
  output:
      autocorr=config['tmpdir'] + 'corr_seg_sites/sims/{scenario}/corr_sA_Sb_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_autocorr_L{L,\d+}_N{N,\d+}.npz'
  resources:
    mem_mb=10000
  run:
    corr_sim = CorrSegSitesSims()
    for f in tqdm(input):
      corr_sim._load_data(f)
    for i in tqdm(range(20)):
      corr_sim.calc_windowed_seg_sites(chrom=i, L=int(wildcards.L))
    i_s = np.arange(1, int(wildcards.N))
    rec_dist = []
    corr_SA_SB = []
    for i in tqdm(i_s):
      a,b =  corr_sim.autocorr_sA_sB(sep=i)
      rec_dist.append(a)
      corr_SA_SB.append(b)
    rec_dists = np.vstack(rec_dist).astype(np.float32)
    corr_SA_SB = np.vstack(corr_SA_SB).astype(np.float32)
    np.savez_compressed(output.autocorr, corr_SA_SB=corr_SA_SB, rec_dists=rec_dists)


rule estimate_autocorr_sA_sB_sims:
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/sims/{scenario}/corr_sA_Sb_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_autocorr_L{L}_N{N}.npz', scenario=['SerialConstant','TennessenEuropean'], mod_n=1, n_anc=1, length=20, Ne=10000, ta=[0, 10000], L=[1000], N=200)


rule monte_carlo_sasb_sims:
  input:
    expand(config['tmpdir'] + 'hap_copying/hap_panels/{{scenario}}/hap_panel_{{mod_n}}_{{n_anc}}_{{ta}}_{{length}}_Ne_{{Ne}}_{seed}.npz', seed=np.arange(1,21))
  output:
    corr_SASB = config['tmpdir'] + 'corr_seg_sites/sims/{scenario}/corr_sA_Sb_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{full_seed,\d+}.monte_carlo_L{L,\d+}.N{N,\d+}.npz'
  wildcard_constraints:
    scenario='(SerialConstant|TennessenEuropean)',
    mod_n = '1',
    n_anc = '1'
  run:
    corr_sim = CorrSegSitesSims()
    for f in tqdm(input):
      corr_sim._load_data(f)
    for i in tqdm(range(20)):
      corr_sim.calc_windowed_seg_sites(chrom=i, L=int(wildcards.L))
    for i in tqdm(range(20)):
      corr_sim.monte_carlo_corr_SA_SB(L=int(wildcards.N), nreps=5000, chrom=i, seed=int(wildcards.full_seed))

    # output slightly less
    rec_rate_mean, rec_rate_se, corr_s1_s2, se_r = corr_sim.gen_binned_rec_rate(bins='auto', range=(5e-6,1e-3))
    np.savez_compressed(output.corr_SASB,
                        scenario=wildcards.scenario,
                        seed=np.int32(wildcards.full_seed),
                        ta = np.int32(wildcards.ta),
                        L = np.int32(wildcards.L),
                        N = np.int32(wildcards.N),
                        rec_rate_mean=rec_rate_mean,
                        rec_rate_se=rec_rate_se,
                        corr_s1_s2=corr_s1_s2,
                        se_r=se_r)



# Landing rule for generating the simulations ...
rule estimate_monte_carlo_sA_sB_sims:
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/sims/{scenario}/corr_sA_Sb_1_1_{ta}_20_Ne_10000_{full_seed}.monte_carlo_L{L}.N{N}.npz', scenario=['SerialConstant','TennessenEuropean'], ta=[0,1000, 10000], L=1000, N=200, full_seed=42),
    expand(config['tmpdir'] + 'corr_seg_sites/sims/{scenario}/corr_sA_Sb_1_1_{ta}_20_Ne_6958_{full_seed}.monte_carlo_L{L}.N{N}.npz', scenario=['SerialConstant'], ta=[0,1000, 10000], L=1000, N=200, full_seed=24)



rule monte_carlo_sA_sB_results:
  input:
    files=rules.estimate_monte_carlo_sA_sB_sims.input
  output:
    'results/corr_seg_sites/monte_carlo_sims_sA_sB_demography.csv'
  run:
    tot_df = []
    for x in tqdm(input.files):
      sim = np.load(x)
      # Loading the specific entries
      scenario = sim['scenario']
      seed = sim['seed']
      N = sim['N']
      ta = sim['ta']
      L = sim['L']
      rec_rate_mean = sim['rec_rate_mean']
      rec_rate_se = sim['rec_rate_se']
      corr_s1_s2 = sim['corr_s1_s2']
      se_r = sim['se_r']
      # Do some light assertions
      assert(rec_rate_mean.size == rec_rate_se.size)
      assert(rec_rate_mean.size == corr_s1_s2.size)
      for i in range(rec_rate_mean.size):
        cur_row = [scenario, N, ta, L, rec_rate_mean[i], rec_rate_se[i], corr_s1_s2[i], se_r[i], seed]
        tot_df.append(cur_row)

    # generate the full output
    final_df = pd.DataFrame(tot_df, columns=['scenario','N','ta','L','rec_rate_mean','rec_rate_se','corr_s1_s2','se_corr','seed'])
    final_df = final_df.dropna()
    final_df.to_csv(str(output), index=False, header=final_df.columns)




### --------- 2. Attempt to estimate parameters ----------- ###

def stack_ragged(array_list, axis=0):
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return(stacked, idx)

rule est_Ne_ta_1kb_sim_bootstrap_monte_carlo_loco:
  """
    Estimating the effective population size
      and the age of the sample from correlation in segregating sites
  """
  input:
    expand(config['tmpdir'] + 'hap_copying/hap_panels/{{scenario}}/hap_panel_{{mod_n}}_{{n_anc}}_{{ta}}_{{length}}_Ne_{{Ne}}_{seed}.npz', seed=np.arange(1,21))
  output:
    bootstrap_params=config['tmpdir'] + 'corr_seg_sites/est_ta/sims/{scenario}/corr_sA_Sb_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{full_seed,\d+}.monte_carlo_L{L}.N{N,\d+}.loco.npz'
  wildcard_constraints:
    scenario='(SerialConstant|TennessenEuropean)',
    mod_n = '1',
    n_anc = '1',
    L = 1000,
    length=20
  run:
    # 1. Generating the correlation object here
    corr_sim = CorrSegSitesSims()
    for f in tqdm(input):
      corr_sim._load_data(f)
    for i in tqdm(range(20)):
      corr_sim.calc_windowed_seg_sites(chrom=i, L=int(wildcards.L))
    for i in tqdm(range(20)):
      corr_sim.monte_carlo_corr_SA_SB(L=int(wildcards.N), nreps=5000, chrom=i, seed=int(wildcards.full_seed))

    # TODO : should we store the intermediates here
    # Leaving one chromosome out, out of 20
    popt_reps = np.zeros(shape=(20,2))
    rec_rate_mean_storage = []
    corr_s1_s2_storage = []

    # Actually running leave-one-chromosome out for testing
    chroms = np.arange(20)
    for i in tqdm(range(20)):
      chroms_subsampled = np.delete(chroms, i, 0)

      rec_rates_mean, _, corr_s1_s2, _ = corr_sim.gen_binned_rec_rate(chroms=chroms_subsampled, bins='auto', range=(1e-5,5e-3))
      #Filter out the nan estimates here
      # TODO : should we put in the weight here
      non_nan_recs = ~np.isnan(rec_rates_mean)
      non_nan_corr = ~np.isnan(corr_s1_s2)
      rec_rates_mean = rec_rates_mean[non_nan_recs]
      corr_s1_s2 = corr_s1_s2[non_nan_corr]
      curpopt,_ = fit_constant_1kb(rec_rates_mean, corr_s1_s2, bounds=(0,[1e8,1e6]))
      popt_reps[i,:] = curpopt
      rec_rate_mean_storage.append(rec_rates_mean)
      corr_s1_s2_storage.append(corr_s1_s2)
    # stacking the arrays
    stacked_rec_rates_mean, idx_rec_rates = stack_ragged(rec_rate_mean_storage)
    stacked_corr_s1_s2_mean, idx_corr_s1_s2 = stack_ragged(corr_s1_s2_storage)

    np.savez_compressed(output.bootstrap_params, scenario=wildcards.scenario, seed=np.int32(wildcards.full_seed), ta = np.int32(wildcards.ta), est_params=popt_reps, rec_rates_mean=stacked_rec_rates_mean, idx_rec_rates=idx_rec_rates, corr_s1_s2=stacked_corr_s1_s2_mean, idx_corr_s1_s2=idx_corr_s1_s2)


rule est_Ne_ta_1kb_sim_final:
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/est_ta/sims/{scenario}/corr_sA_Sb_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{full_seed}.monte_carlo_L{L}.N{N}.loco.npz', full_seed=42, Ne=10000, length=20, ta=[0,100,1000,10000], mod_n=1, n_anc=1, scenario=['SerialConstant', 'TennessenEuropean'], L=1000, N=200)

rule collapse_est_ta_Ne:
  input:
    files = rules.est_Ne_ta_1kb_sim_final.input
  output:
    'results/corr_seg_sites/est_ta_Ne_corr_sa_sb.csv'
  run:
    tot_df_rows = []
    for f in tqdm(input.files):
      cur_df = np.load(f)
      scenario = cur_df['scenario']
      ta = cur_df['ta']
      est_params = cur_df['est_params']
      Ne_est = est_params[:,0]
      ta_est = est_params[:,1]

      mean_Ne = np.mean(Ne_est)
      se_Ne = np.std(Ne_est, ddof=1)/ np.sqrt(Ne_est.size)
      mean_ta = np.mean(ta_est)
      se_ta = np.std(ta_est, ddof=1) / np.sqrt(ta_est.size)
      cur_row = [scenario, ta, mean_ta, se_ta, mean_Ne, se_Ne]
      tot_df_rows.append(cur_row)
    final_df = pd.DataFrame(tot_df_rows, columns=['scenario','ta','ta_est','se_ta_est', 'Ne_est', 'se_Ne_est'])
    final_df = final_df.dropna()
    final_df.to_csv(str(output), index=False, header=final_df.columns)


# -------------- 3. Estimation from Real Data --------------- #
# Individuals who we want for our ancient samples
anc_indivs = ['Loschbour', 'LBK','Bichon', 'UstIshim']

rec_bins = np.arange(0.01, 1.01, 0.01)
rec_map_types = {"Physical_Pos" : np.uint64,
                 "deCODE" : np.float32,
                 "COMBINED_LD" : np.float32,
                 "YRI_LD": np.float32,
                 "CEU_LD" : np.float32,
                 "AA_Map": np.float32,
                 "African_Enriched" : np.float32,
                 "Shared_Map" : np.float32}


# #Directory of current data currently
# data_dir = '/home/abiddanda/novembre_lab2/old_project/share/botai/data/vcfs/'
# sgdp_dir = '/home/abiddanda/novembre_lab2/data/external_public/sgdp/merged/'
# thousand_genomes_dir = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/haps/'
# KGP_BIALLELIC = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.biallelic_snps.vcf.gz'

# rule kg_phase3_ancient_merge:
#   input:
#     merged_vcf = 'data/raw_data/merged_kgp/ancient_kgp_total_merged.chr{CHROM}.vcf.gz',
#     merged_vcf_idx = 'data/raw_data/merged_kgp/ancient_kgp_total_merged.chr{CHROM}.vcf.gz.tbi'

rule gen_seg_sites_table_haploid_modern_test:
  """
    Calculate table of segregating sites when we have a modern haplotype
  """
  input:
    vcf = 'data/raw_data/merged_wgs_ancient/ancient_kgp_total_merged.chr{CHROM}.vcf.gz',
    rec_df = 'data/raw_data/full_recomb_maps/maps_chr.{CHROM}'
  output:
    tmp_vcf = temp(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.haploid_modern_{hap,\d+}.vcf.gz'),
    tmp_vcf_idx = temp(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.haploid_modern_{hap,\d+}.vcf.gz.tbi'),
    pos_ac = config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.haploid_modern_{hap,\d+}.test.npz'
  wildcard_constraints:
    recmap = '(COMBINED_LD|deCODE)',
    proj = 'kgp'
  run:
    # 1. Generate the filtered VCF here ...
    indiv_ids = "%s,%s" % (wildcards.ANC, wildcards.MOD)
    shell('bcftools view -s {indiv_ids} -c 1:minor {input.vcf} | bgzip -@10 > {output.tmp_vcf}')
    shell('tabix -f {output.tmp_vcf}')
    # 2. Look and read in the data set
    vcf_data = allel.read_vcf(output.tmp_vcf)
    gt = vcf_data['calldata/GT']
    pos = vcf_data['variants/POS']
    chrom = vcf_data['variants/CHROM']
    gt_anc = gt[:,0,0] + gt[:,0,1]
    gt_mod_hap = gt[:,1,int(wildcards.hap)]

    # generate recombination rates here
    rec_df = pd.read_csv(input.rec_df, sep='\s+', low_memory=True, dtype=rec_map_types)
    rec_pos = np.array(rec_df['Physical_Pos'], dtype=np.uint32)
    rec_dist = np.array(rec_df[str(wildcards.recmap)], dtype=np.float32)
    interp_rec_pos = np.interp(pos, rec_pos, rec_dist)
    np.savez_compressed(output.pos_ac, chrom=chrom, pos=pos, gt_anc=gt_anc, gt_mod_hap=gt_mod_hap, rec_pos=interp_rec_pos)

rule gen_seg_sites_table_diploid_test:
  """
    Calculates segregating sites within diploid samples in a way similar to the haploid version above
  """
  input:
    vcf = 'data/raw_data/merged_wgs_ancient/ancient_kgp_total_merged.chr{CHROM}.vcf.gz',
    rec_df = 'data/raw_data/full_recomb_maps/maps_chr.{CHROM}'
  output:
    tmp_vcf = temp(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.diploid.vcf.gz'),
    tmp_vcf_idx = temp(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.diploid.vcf.gz.tbi'),
    pos_ac = config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.diploid.test.npz'
  wildcard_constraints:
    recmap = '(COMBINED_LD|deCODE)',
    proj = 'kgp'
  run:
    indiv_ids = "%s,%s" % (wildcards.ANC, wildcards.MOD)
    shell('bcftools view -s {indiv_ids} -c 1:minor {input.vcf} | bgzip > {output.tmp_vcf}; tabix -f {output.tmp_vcf}')
    vcf_data = allel.read_vcf(output.tmp_vcf)
    gt = vcf_data['calldata/GT']
    pos = vcf_data['variants/POS']
    chrom = vcf_data['variants/CHROM']
    gt_anc = gt[:,0,0] + gt[:,0,1]
    gt_mod = gt[:,1,0] + gt[:,1,1]
    # generate recombination rates here
    rec_df = pd.read_csv(input.rec_df, sep='\s+', low_memory=True, dtype=rec_map_types)
    rec_pos = np.array(rec_df['Physical_Pos'], dtype=np.uint32)
    rec_dist = np.array(rec_df[str(wildcards.recmap)], dtype=np.float32)
    interp_rec_pos = np.interp(pos, rec_pos, rec_dist)
    np.savez_compressed(output.pos_ac, chrom=chrom, pos=pos, gt_anc=gt_anc, gt_mod=gt_mod, rec_pos=interp_rec_pos)


rule haploid_modern_test:
  """
    Generating the haploid modern testing setup
  """
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.haploid_modern_{hap}.test.npz', ANC=['LBK', 'Bichon', 'Loschbour', 'UstIshim', 'NA12878'], MOD=['NA12830'], recmap='deCODE', proj='kgp', hap=[1], CHROM=np.arange(1,23))


rule diploid_modern_test:
  """
    Generating the diploid testing setup
  """
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{ANC}_mod_{MOD}/interpolated_{proj}_{recmap}_chr{CHROM}.diploid.test.npz', CHROM=np.arange(1,23), ANC=['LBK', 'UstIshim'], MOD=['NA12830'], recmap='deCODE', proj='kgp')


PILOT_MASK = 'data/raw_data/genome_masks/pilot_exclusion_mask.renamed.bed'
CENTROMERE_MASK = 'data/raw_data/genome_masks/hg19.centromere.autosomes.renamed.bed'
CENTROMERE_AND_MAPPABILITY = 'data/raw_data/genome_masks/exclusion_mask/hs37d5_autosomes.mask.exclude.filt.bed'

MASK_DICT = {'none' : None, 'pilot' : PILOT_MASK, 'centromere': CENTROMERE_MASK, 'centromere_mappability': CENTROMERE_AND_MAPPABILITY}

rule monte_carlo_sA_sB_est_haploid_autosomes:
  """
    Estimate of the Monte-Carlo correlation in sA  vs. sB for all chromosomes
  """
  input:
    seg_sites_haploid = expand(config['tmpdir'] + 'corr_seg_sites/real_data/anc_{{ANC}}_mod_{{MOD}}/interpolated_{{proj}}_{{recmap}}_chr{CHROM}.haploid_modern_{{hap}}.test.npz', CHROM=np.arange(1,23)),
    mask =  lambda wildcards: MASK_DICT[wildcards.mask]
  output:
    corr_SASB = config['tmpdir'] + 'corr_seg_sites/monte_carlo_results/anc_{ANC}_mod_{MOD}/autosomes.paired_seg_sites.{proj}.{recmap}.{mask}.hap{hap,\d+}.seed{seed,\d+}.monte_carlo_L{L,\d+}.N{N,\d+}.npz'
  wildcard_constraints:
    recmap = '(COMBINED_LD|deCODE)',
    mask='(centromere|centromere_mappability|pilot)',
    proj='kgp',
    hap='0|1'
  run:
    cur_corr = CorrSegSitesRealDataHaploid()
    for f in tqdm(input.seg_sites_haploid):
      cur_corr._load_data(f, miss_to_nan=False)
    cur_corr._conv_cM_to_morgans()
    for i in tqdm(cur_corr.chrom_pos_dict):
      cur_corr.calc_windowed_seg_sites(chrom=i, mask=input.mask)
      cur_corr.monte_carlo_corr_SA_SB(L=int(wildcards.N), nreps=2000, chrom=i, seed=int(wildcards.seed))
    # Computing the monte-carlo estimates that we have
    xs = np.logspace(-5.0, -3, 30)
    rec_rate_mean, rec_rate_se, corr_s1_s2, se_r = cur_corr.gen_binned_rec_rate(bins=xs)
    np.savez_compressed(output.corr_SASB, rec_rate_mean=rec_rate_mean, rec_rate_se=rec_rate_se, corr_s1_s2=corr_s1_s2, se_r=se_r)


# Testing out with multiple samples in there ...
pop_1kg_df = pd.read_table('data/raw_data/1kg_panel/integrated_call_samples_v3.20130502.ALL.panel')

test_yri_1kg = pop_1kg_df[pop_1kg_df['pop'] == 'YRI']['sample'].values[:3]
test_chb_1kg = pop_1kg_df[pop_1kg_df['pop'] == 'CHB']['sample'].values[:3]
test_ceu_1kg = pop_1kg_df[pop_1kg_df['pop'] == 'CEU']['sample'].values
test_gih_1kg = pop_1kg_df[pop_1kg_df['pop'] == 'GIH']['sample'].values[:3]
test_indivs_1kg = np.hstack([test_yri_1kg, test_chb_1kg, test_ceu_1kg, test_gih_1kg])

# np.array('NA12830')

rule monte_carlo_real_data_1kg_samples:
  input:
    expand(config['tmpdir'] + 'corr_seg_sites/monte_carlo_results/anc_{ANC}_mod_{MOD}/autosomes.paired_seg_sites.{proj}.{recmap}.{mask}.hap{hap}.seed{seed}.monte_carlo_L{L}.N{N}.npz', ANC=['LBK', 'UstIshim'], MOD=test_indivs_1kg, recmap='deCODE', mask=['centromere'], proj='kgp', hap=[1], seed=42,  L=1000, N=[50])


rule concatenate_tot_corr_piA_piB:
  input:
    files = expand(config['tmpdir'] + 'corr_seg_sites/monte_carlo_results/anc_{ANC}_mod_{MOD}/autosomes.paired_seg_sites.{proj}.{recmap}.{mask}.hap{hap}.seed{seed}.monte_carlo_L{L}.N{N}.npz', ANC=['LBK', 'UstIshim'], MOD=test_indivs_1kg, recmap='deCODE', mask=['centromere'], proj='kgp', hap=[1], seed=42,  L=1000, N=50)
  output:
    'results/corr_seg_sites/monte_carlo_est_LBK_UstIshim.csv'
  run:
    tot_df_rows = []
    for a in ['LBK','UstIshim']:
      # Getting only files with that filename on them
      valid_files = [x for x in input.files if a in x]
      for x in tqdm(test_indivs_1kg):
        print('Corr(pi_B, pi_B) with: %s' % x)
        fname = [y for y in valid_files if x in y]
        assert(len(fname) >= 1)
        print(fname)
        df = np.load(fname[-1])
        rec_rate_mean = df['rec_rate_mean']
        mean_corr_s1_s2 = df['corr_s1_s2']
        rec_rate_se = df['rec_rate_se']
        se_r = df['se_r']
        assert(rec_rate_mean.size == mean_corr_s1_s2.size)
        for i in range(rec_rate_mean.size):
          cur_row = [a,x,rec_rate_mean[i], rec_rate_se[i], mean_corr_s1_s2[i], se_r[i]]
          tot_df_rows.append(cur_row)
    df = pd.DataFrame(tot_df_rows, columns=['ANC_ID','MOD_ID','rec_rate_mean','rec_rate_se', 'corr_piA_piB', 'se_corr_piA_piB'])
    final_df = df.dropna()
    final_df.to_csv(str(output), index=False, header=final_df.columns)
