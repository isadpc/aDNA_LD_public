#!python3

import os
import sys
import numpy as np
import msprime as msp
import tszip
import allel
import pandas as pd
from tqdm import tqdm

sys.path.append('src/')
from li_stephens import LiStephensHMM
from aDNA_coal_sim import *

# Import configurations to add some things
configfile: "config.yml"


### --- Simulation Parameters --- ###
lengths = [5,10,20]
rec_rate = 1e-8
mut_rate = 1e-8
Ne = 10000
nreps=20

# Data file for IBDNe UK10K demography
ibd_ne_demo_file = 'data/demo_models/uk10k.IBDNe.txt'

def ascertain_variants(hap_panel, pos, maf=0.05):
    """ Ascertaining variants based on frequency """
    assert((maf < 0.5) & (maf > 0))
    mean_daf = np.mean(hap_panel, axis=0)
    idx = np.where((mean_daf > maf) | (mean_daf < (1. - maf)))[0]
    asc_panel = hap_panel[:,idx]
    asc_pos = pos[idx]
    return(asc_panel, asc_pos, idx)



###### --------- Simulations ----------- ######
# 'data/hap_copying/hap_panels/{{scenario}}/hap_panel_{{mod_n}}_{{n_anc}}_{{ta}}_{{length}}_Ne_{{Ne}}_{rep}.npz'

rule sim_demography_single_ancients:
  """
    Generate a tree-sequence of multiple ages
  """
  output:
    treeseq= config['tmpdir'] + 'hap_copying/hap_panels/{scenario}/hap_panel_{mod_n,\d+}_{n_anc}_{ta,\d+}_{length,\d+}_Ne_{Ne,\d+}_{rep,\d+}.treeseq.gz',
    hap_panel=config['tmpdir'] + 'hap_copying/hap_panels/{scenario}/hap_panel_{mod_n,\d+}_{n_anc}_{ta,\d+}_{length,\d+}_Ne_{Ne,\d+}_{rep,\d+}.npz',
  wildcard_constraints:
    scenario = '(SerialConstant|TennessenEuropean|IBDNeUK10K)'
  run:
    # Setup the parameters (length in megabases)
    length=np.float32(wildcards.length) * 1e6
    Ne = np.int32(wildcards.Ne)
    ta = np.int32(wildcards.ta)
    mod_n = np.int32(wildcards.mod_n)
    n_anc = np.int32(wildcards.n_anc)
    scenario = wildcards.scenario
    cur_sim = None
    if scenario == 'SerialConstant':
      cur_sim = SerialConstant(Ne=Ne, mod_n=mod_n, t_anc=[ta], n_anc=[n_anc])
    elif scenario == 'TennessenEuropean':
      cur_sim = SerialTennessenModel()
      cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=[n_anc], t_anc=[ta])
    elif scenario == 'IBDNeUK10K':
      cur_sim = SerialIBDNeUK10K(demo_file=ibd_ne_demo_file)
      cur_sim._set_demography()
      cur_sim._add_samples(n_mod = mod_n, n_anc=[n_anc], t_anc=[ta])
    else:
      raise ValueError('Improper value input for this simulation!')
    # Conducting the actual simulation ...
    ts = cur_sim._simulate(mutation_rate=mut_rate, recombination_rate=rec_rate, length=length)
    tszip.compress(ts, str(output.treeseq))
    # Generating the haplotype reference panel...
    geno = ts.genotype_matrix().T
    phys_pos = np.array([v.position for v in ts.variants()])
    rec_pos = phys_pos * rec_rate
    node_ids = [s for s in ts.samples()]
    tree = ts.first()
    times = np.array([tree.time(x) for x in node_ids])
    np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times)


rule check_corr_seg_sites:
  """
    Estimate the correlation in segregating sites in different demographic scenarios
  """
  input:
    expand(config['tmpdir'] + 'hap_copying/hap_panels/{scenario}/hap_panel_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{rep}.npz', mod_n=1, n_anc=1, rep=np.arange(20), scenario=['SerialConstant', 'TennessenEuropean'], ta=[0, 10000], length=20, Ne=10000)


rule sim_demography_multi_ancients:
  """
    Generate a tree-sequence of multiple ages
  """
  output:
    treeseq=config['tmpdir'] + 'full_sim_all/{scenario}/generations_{ta,\d+}_{interval,\d+}/serial_coal_{mod_n, \d+}_{n_anc, \d+}_{length, \d+}_Ne{Ne,\d+}_{rep,\d+}.treeseq.gz',
  wildcard_constraints:
    scenario = '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialBottleneckInstant[0-9]*|IBDNeUK10K|SimpleGrowth[0-9]*)'
  run:
    # Setup the parameters (length in megabases)
    length=np.float32(wildcards.length) * 1e6
    Ne = np.int32(wildcards.Ne)
    t_a_max = np.int32(wildcards.ta)
    interval = np.int32(wildcards.interval)
    assert(t_a_max % interval == 0)
    mod_n = np.int32(wildcards.mod_n)
    n_a = np.int32(wildcards.n_anc)
    # Note we need to add in one so that we have an inclusive set
    t_anc = np.arange(interval, t_a_max+1, interval).tolist()
    n_anc = np.repeat(n_a, len(t_anc)).tolist()
    scenario = wildcards.scenario
    cur_sim = None
    if scenario == 'SerialConstant':
      cur_sim = SerialConstant(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc)
    elif scenario == 'TennessenEuropean':
      cur_sim = SerialTennessenModel()
      cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    elif scenario == 'SerialBottleneck':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc)
    elif scenario == 'SerialBottleneckLate':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=1000, bottle_duration=500)
    elif scenario == 'SerialBottleneckInstant1':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=100, bottle_duration=500000, bottle_mag=0.01)
    elif scenario == 'SerialBottleneckInstant2':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=200, bottle_duration=500000, bottle_mag = 0.01)
    elif scenario == 'SerialBottleneckInstant3':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=400, bottle_duration=500000, bottle_mag = 0.01)
    elif scenario == 'SerialBottleneckInstant4':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=100, bottle_duration=500000, bottle_mag=0.001)
    elif scenario == 'SerialBottleneckInstant5':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=200, bottle_duration=500000, bottle_mag = 0.001)
    elif scenario == 'SerialBottleneckInstant6':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=400, bottle_duration=500000, bottle_mag = 0.001),
    elif scenario == 'SerialBottleneckInstant7':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=100, bottle_duration=500000, bottle_mag=0.0001)
    elif scenario == 'SerialBottleneckInstant8':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=200, bottle_duration=500000, bottle_mag = 0.0001)
    elif scenario == 'SerialBottleneckInstant9':
      cur_sim = SerialBottleneck(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc, bottle_start=400, bottle_duration=500000, bottle_mag = 0.0001)
    elif scenario == 'IBDNeUK10K':
      cur_sim = SerialIBDNeUK10K(demo_file=ibd_ne_demo_file)
      cur_sim._set_demography()
      cur_sim._add_samples(n_mod = mod_n, n_anc = n_anc, t_anc = t_anc)
    elif scenario == 'SimpleGrowth1':
      cur_sim = SerialGrowthSimple(r=1e-3)
      cur_sim._add_samples(n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    elif scenario == 'SimpleGrowth2':
      cur_sim = SerialGrowthSimple(r=5e-3)
      cur_sim._add_samples(n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    elif scenario == 'SimpleGrowth3':
      cur_sim = SerialGrowthSimple(r=1e-2)
      cur_sim._add_samples(n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    elif scenario == 'SimpleGrowth4':
      cur_sim = SerialGrowthSimple(r=2.5e-2)
      cur_sim._add_samples(n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    else:
      raise ValueError('Improper value input for this simulation!')
    # Conducting the actual simulation ...
    tree_seq = cur_sim._simulate(mutation_rate=mut_rate, recombination_rate=rec_rate, length=length)
    tszip.compress(tree_seq, str(output.treeseq))

rule create_hap_panel_all:
  """
    Creates a haplotype panel for a joint simulation
  """
  input:
    treeseq = rules.sim_demography_multi_ancients.output.treeseq
  output:
    hap_panel = config['tmpdir']+'full_sim_all/{scenario}/generations_{ta,\d+}_{interval,\d+}/hap_panel_{mod_n, \d+}_{n_anc, \d+}_{length, \d+}_Ne{Ne,\d+}_{rep,\d+}.panel.npz'
  run:
    ts = tszip.decompress(input.treeseq)
    geno = ts.genotype_matrix().T
    phys_pos = np.array([v.position for v in ts.variants()])
    rec_pos = phys_pos * rec_rate
    node_ids = [s for s in ts.samples()]
    tree = ts.first()
    times = np.array([tree.time(x) for x in node_ids])
    np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times)


rule infer_scale_serial_all_ascertained:
  """
    Infer scale parameter using a naive Li-Stephens Model
      and ascertaining to snps in the modern panel at a high-frequency
    NOTE : should we have considerations for
  """
  input:
    hap_panel = rules.create_hap_panel_all.output.hap_panel
  wildcard_constraints:
    scenario = '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialBottleneckInstant[0-9]*|SerialMigration_[0-9]*|IBDNeUK10K|SimpleGrowth[0-9]*)'
  output:
    mle_hap_est = config['tmpdir'] + 'hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n, \d+}_{n_anc, \d+}_{length,\d+}_Ne{Ne,\d+}_{rep,\d+}.asc_{asc, \d+}.ta_{ta_samp, \d+}.scale.npz'
  run:
    # loading in the data
    cur_data = np.load(input.hap_panel)
    hap_panel_test = cur_data['haps']
    pos = cur_data['rec_pos']
    times = cur_data['ta']
    times = times.astype(np.float32)
    # Testing things out here ...
    mod_idx = np.where(times == 0)[0]
    ta_test = np.float32(wildcards.ta_samp)
    ta_idx = np.where(times == ta_test)[0][0]
    # Extracting the panel
    modern_hap_panel = hap_panel_test[mod_idx,:]
    # Test haplotype in simulations is always the last haplotype...
    test_hap = hap_panel_test[ta_idx,:]
    mod_asc_panel, asc_pos, asc_idx = ascertain_variants(modern_hap_panel, pos, maf = np.int32(wildcards.asc)/100.)
    anc_asc_hap = test_hap[asc_idx]

    afreq_mod = np.sum(mod_asc_panel, axis=1)

    cur_hmm = LiStephensHMM(haps = mod_asc_panel, positions=asc_pos)
    scales = np.logspace(2,4,20)
    neg_log_lls = np.array([cur_hmm.negative_logll(anc_asc_hap, scale=s, eps=1e-2) for s in tqdm(scales)])
    mle_scale = cur_hmm.infer_scale(anc_asc_hap, eps=1e-2, method='Bounded', bounds=(1.,1e6), tol=1e-4)
    # Estimating both error and scale parameters jointly
    mle_params = cur_hmm.infer_params(anc_asc_hap, x0=[1e2,1e-4], bounds=[(1e1,1e7), (1e-6,1e-1)], tol=1e-4)
    cur_params = np.array([np.nan, np.nan])
    se_params = np.array([np.nan, np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
      se_params = np.array([np.sqrt(mle_params.hess_inv.todense()[0,0]), np.sqrt(mle_params.hess_inv.todense()[1,1])])
    model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
    # NOTE: we might need some different params being inferred ... 
    np.savez(output.mle_hap_est,
             scenario=wildcards.scenario,
             Ne=np.int32(wildcards.Ne),
             scales=scales, 
             loglls=-neg_log_lls, 
             scale=mle_scale['x'], 
             params=cur_params, 
             se_params=se_params, 
             model_params=model_params, 
             mod_freq = afreq_mod)


# NOTE : we should keep the same time intervals for sampling here ...
rule calc_infer_scales_asc_all_figures:
  """Rule to calculate all of the simulations for figures on haplotype copying models as a function of time
  """
  input:
    expand(config['tmpdir'] + 'hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialConstant'], ta=1000, interval=10, mod_n=100, n_anc=1, length=40, Ne=[20000, 10000,5000], rep=np.arange(5), asc=5, ta_samp=np.arange(10,501,10)),
    expand(config['tmpdir'] + 'hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['IBDNeUK10K', 'TennessenEuropean'], ta=1000, interval=10, mod_n=100, n_anc=1, length=40, Ne=[10000], rep=np.arange(5), asc=5, ta_samp=np.arange(10,501,10)),
    expand(config['tmpdir'] + 'hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialBottleneckInstant7', 'SerialBottleneckInstant8', 'SerialBottleneckInstant9'], ta=400, interval=10, mod_n=100, n_anc=1, length=40, Ne=1000000, rep=np.arange(5), asc=5, ta_samp=np.arange(10,401,10)),
#     expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SimpleGrowth1', 'SimpleGrowth2', 'SimpleGrowth3', 'SimpleGrowth4'], ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=1000000, rep=np.arange(5), asc=5, ta_samp=np.arange(5,401,5))

rule concatenate_hap_copying_results:
  input:
    files = rules.calc_infer_scales_asc_all_figures.input
  output:
    'results/hap_copying/simulations/jump_rate_est_sims.csv'
  run:
    tot_df_rows = []
    for f in tqdm(input.files):
      cur_df = np.load(f)
      #Extract the relevant parameters
      scenario = cur_df['scenario']
      N = cur_df['Ne']
      scale = cur_df['scale']
      params = cur_df['params']
      se_params = cur_df['se_params']
      model_params = cur_df['model_params']
      cur_row = [scenario, N, scale, params[0],params[1], se_params[0],se_params[1], model_params[0], model_params[1], model_params[2]]
      tot_df_rows.append(cur_row)
    final_df = pd.DataFrame(tot_df_rows, columns=['scenario','Ne', 'scale_marginal','scale_jt', 'eps_jt','se_scale_jt', 'se_eps_jt','n_panel','n_snps','ta'])
    # Concatenate to create a new dataframe
    final_df.to_csv(str(output), index=False, header=final_df.columns)
      



      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
# # ---------- Running a serial simulation according to the real Reich Lab Data  -----------#

# gen_time = 30

# # TODO : should we simulate a specific chromosome?
# rule sim_serial_data_1kg_ceu_real_chrom:
#   """
#     Simulating example data for the CEU setting
#   """
#   input:
#     recmap = rules.sim_format_recombination_map.output.recmap_hapmap_format,
#     times_ceu_km = lambda wildcards: 'data/hap_copying/time_points/ceu_%dkm_ages.npy' % int(wildcards.km),

#   output:
#     treeseq = 'data/full_sim_all/{scenario}/ceu_sim_{km, \d+}_{CHROM}_{genmap}/serial_coal_{mod_n, \d+}_Ne{Ne,\d+}_{rep,\d+}.treeseq.gz'
#   wildcard_constraints:
#     scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)'
#   run:
#     times_kya = np.load(input.times_ceu_km)
#     times_gen = np.array(times_kya / gen_time, dtype=np.int32)
#     unique_times, n_samples = np.unique(times_gen, return_counts=True)
#     recmap = msp.RecombinationMap.read_hapmap(input.recmap)
#     Ne = np.int32(wildcards.Ne)
#     mod_n = np.int32(wildcards.mod_n)
#     t_anc = unique_times.tolist()
#     n_anc = n_samples.tolist()
#     scenario = wildcards.scenario
#     cur_sim = None
#     if scenario == 'SerialConstant':
#       cur_sim = SerialConstant(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc)
#     elif scenario == 'TennessenEuropean':
#       cur_sim = SerialTennessenModel()
#       cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
#     elif scenario == 'TennessenDoubleGrowthEuropean':
#       cur_sim = SerialTennessenModel()
#       cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
#       cur_sim._change_growth_rate(r_EU=0.039)
#     elif scenario == 'TennessenQuadGrowthEuropean':
#       cur_sim = SerialTennessenModel()
#       cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
#       cur_sim._change_growth_rate(r_EU=0.078)
#     elif scenario == 'IBDNeUK10K':
#       cur_sim = SerialIBDNeUK10K(demo_file=ibd_ne_demo_file)
#       cur_sim._set_demography()
#       cur_sim._add_samples(n_mod = mod_n, n_anc = n_anc, t_anc = t_anc)
#     else:
#       raise ValueError('Improper scenario input for this simulation!')
#     # Conducting the actual simulation ...
#     tree_seq = cur_sim._simulate(mutation_rate=mut_rate, recombination_map=recmap)
#     tszip.compress(tree_seq, str(output.treeseq))

# rule create_hap_panel_1kg_ceu_real_chrom:
#   """
#     Creates a haplotype panel for a joint simulation
#   """
#   input:
#     treeseq = rules.sim_serial_data_1kg_ceu_real_chrom.output.treeseq
#   wildcard_constraints:
#     scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)',
#     genmap = '(deCODE|COMBINED_LD)'
#   output:
#     hap_panel = 'data/full_sim_all/{scenario}/ceu_sim_{km, \d+}_{CHROM}_{genmap}/serial_coal_{mod_n, \d+}_Ne{Ne,\d+}_{rep,\d+}.panel.npz'
#   run:
#     ts = tszip.decompress(input.treeseq)
#     geno = ts.genotype_matrix().T
#     phys_pos = np.array([v.position for v in ts.variants()])
#     rec_pos = phys_pos * rec_rate
#     node_ids = [s for s in ts.samples()]
#     tree = ts.first()
#     times = np.array([tree.time(x) for x in node_ids])
#     np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times)


# rule infer_scale_serial_ascertained_ceu_sims:
#   """
#     Infer scale parameter using a naive Li-Stephens Model
#       and ascertaining to snps in the modern panel at
#   """
#   input:
#     hap_panel = rules.create_hap_panel_1kg_ceu_real_chrom.output.hap_panel,
#     times_ceu_km = lambda wildcards: 'data/hap_copying/time_points/ceu_%dkm_ages.npy' % int(wildcards.km)
#   wildcard_constraints:
#     scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)',
#     genmap = '(deCODE|COMBINED_LD)'
#   output:
#     mle_hap_est = 'data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km,\d+}_{CHROM}_{genmap}/mle_scale_{mod_n, \d+}_Ne{Ne,\d+}_{rep, \d+}.asc_{asc, \d+}.ta_{ta_samp, \d+}.scale.npz'
#   run:
#     # loading in the data
#     cur_data = np.load(input.hap_panel)
#     hap_panel_test = cur_data['haps']
#     pos = cur_data['rec_pos']
#     times = cur_data['ta']
#     times = times.astype(np.float32)
#     # Testing things out here ...
#     mod_idx = np.where(times == 0)[0]
#     ta_test = np.float32(wildcards.ta_samp)
#     ta_idx = np.where(times == ta_test)[0][0]
#     # Extracting the panel
#     modern_hap_panel = hap_panel_test[mod_idx,:]
#     # Test haplotype in simulations is always the last haplotype...
#     test_hap = hap_panel_test[ta_idx,:]
#     mod_asc_panel, asc_pos, asc_idx = ascertain_variants(modern_hap_panel, pos, maf = np.int32(wildcards.asc)/100.)
#     anc_asc_hap = test_hap[asc_idx]
#     afreq_mod = np.sum(mod_asc_panel, axis=1)
#     cur_hmm = LiStephensHMM(haps = mod_asc_panel, positions=asc_pos)
#     # Setting theta here ...
#     cur_hmm.theta = cur_hmm._infer_theta()
#     # Setting the error rate to be similar to the original LS-Model
#     eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
#     mle_scale = cur_hmm._infer_scale(anc_asc_hap, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
#     # Estimating both error and scale parameters jointly
#     mle_params = cur_hmm._infer_params(anc_asc_hap, x0=[1e2,1e-3], bounds=[(1.,1e7), (1e-6,0.9)], tol=1e-7)
#     cur_params = np.array([np.nan, np.nan])
#     if mle_params['success']:
#       cur_params = mle_params['x']
#     model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
#     np.savez(output.mle_hap_est, scale=mle_scale['x'], params=cur_params, model_params=model_params, mod_freq = afreq_mod)


# # # Actually loading in some of these schemes
# # test_1500_km = np.load('data/hap_copying/time_points/ceu_1500km_ages.npy')/gen_time
# # test_1500_km = np.unique(test_1500_km.astype(int))

# # test_2500_km = np.load('data/hap_copying/time_points/ceu_2500km_ages.npy')/gen_time
# # test_2500_km = np.unique(test_2500_km.astype(int))

# # rule test_ceu_infer_scale_real_chrom:
# #   input:
# # #     expand('data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km}_{CHROM}_deCODE/mle_scale_{mod_n}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz',scenario=['SerialConstant', 'TennessenEuropean', 'TennessenDoubleGrowthEuropean','TennessenQuadGrowthEuropean'], km=2500, mod_n=49, CHROM='X', Ne=10000, rep=0, asc=10, ta_samp=test_2500_km),
# #     expand('data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km}_{CHROM}_deCODE/mle_scale_{mod_n}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz',scenario=['SerialConstant', 'TennessenEuropean', 'IBDNeUK10K'], km=1500, mod_n=49, CHROM='X', Ne=10000, rep=0, asc=5, ta_samp=test_1500_km)
