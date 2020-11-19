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
    treeseq=config['tmpdir'] + 'data/full_sim_all/{scenario}/generations_{ta,\d+}_{interval,\d+}/serial_coal_{mod_n, \d+}_{n_anc, \d+}_{length, \d+}_Ne{Ne,\d+}_{rep,\d+}.treeseq.gz',
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
    hap_panel = config['tmpdir']+'data/full_sim_all/{scenario}/generations_{ta,\d+}_{interval,\d+}/hap_panel_{mod_n, \d+}_{n_anc, \d+}_{length, \d+}_Ne{Ne,\d+}_{rep,\d+}.panel.npz'
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
    mle_hap_est = config['tmpdir'] + 'data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n, \d+}_{n_anc, \d+}_{length,\d+}_Ne{Ne,\d+}_{rep,\d+}.asc_{asc, \d+}.ta_{ta_samp, \d+}.scale.npz'
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
    print(times, ta_test)
    ta_idx = np.where(times == ta_test)[0][0]
    print(ta_idx)
    # Extracting the panel
    modern_hap_panel = hap_panel_test[mod_idx,:]
    # Test haplotype in simulations is always the last haplotype...
    test_hap = hap_panel_test[ta_idx,:]
    mod_asc_panel, asc_pos, asc_idx = ascertain_variants(modern_hap_panel, pos, maf = np.int32(wildcards.asc)/100.)
    anc_asc_hap = test_hap[asc_idx]
    
    afreq_mod = np.sum(mod_asc_panel, axis=1)
    
    cur_hmm = LiStephensHMM(haps = mod_asc_panel, positions=asc_pos)
    # Setting theta here ...
    cur_hmm.theta = cur_hmm._infer_theta()
    # Setting the error rate to be similar to the original LS-Model
    eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
    scales = np.logspace(2,6,50)
    neg_log_lls = np.array([cur_hmm._negative_logll(anc_asc_hap, scale=s, eps=eps) for s in tqdm(scales)])
    mle_scale = cur_hmm._infer_scale(anc_asc_hap, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
    # Estimating both error and scale parameters jointly
    mle_params = cur_hmm._infer_params(anc_asc_hap, x0=[1e2,1e-3], bounds=[(1.,1e7), (1e-6,0.1)], tol=1e-7)
    cur_params = np.array([np.nan, np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
    np.savez(output.mle_hap_est, scales=scales, loglls=-neg_log_lls, scale=mle_scale['x'], params=cur_params, model_params=model_params, mod_freq = afreq_mod)    

    
rule test_asc_scale_inf:
  input:
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialConstant','TennessenEuropean','SerialBottleneck','SerialBottleneckLate'], ta=1000, interval=100, mod_n=100, n_anc=1, length=20, Ne=10000, rep=np.arange(5), asc=5, ta_samp=np.arange(100,1001,100))


rule create_hap_panel_test:
  """ Calculate an underlying test haplotype """
  input:
    treeseq = 'data/full_sim/{scenario}/generations/serial_coal_{mod_n, \d+}_{n_anc, \d+}_{ta, \d+}_{length, \d+}_Ne{Ne,\d+}_{rep,\d+}.treeseq.gz'
  output:
    hap_panel = 'data/hap_copying/hap_panels/{scenario}/hap_panel_{mod_n, \d+}_{n_anc, \d+}_{ta, \d+}_{length, \d+}_Ne_{Ne,\d+}_{rep,\d+}.npz'
  wildcard_constraints:
    scenario =
    '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialMigration_[0-9]*|SerialBottleneck_[0-9]*_[0-9]*_[0-9]*)'
  run:
    ts = tszip.decompress(input.treeseq)
    geno = ts.genotype_matrix().T
    # NOTE : here we are generating the underlying positions in morgans
    phys_pos = np.array([v.position for v in ts.variants()])
    rec_pos = phys_pos * rec_rate
    # Getting the times up in here ... 
    tree = ts.first()
    node_ids = [s for s in ts.samples()]
    times = np.array([tree.time(x) for x in node_ids])
    np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times)

    
rule infer_scale_serial:
  """ Infer scale parameter using a naive Li-Stephens Algorithm """
  input:
    hap_panel = rules.create_hap_panel_test.output.hap_panel
  wildcard_constraints:
    scenario = '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialMigration_[0-9]*|SerialBottleneck_[0-9]*_[0-9]*_[0-9]*)'
  output:
    log_ll = 'data/hap_copying/mle_results/{scenario}/mle_scale_{mod_n, \d+}_{n_anc, \d+}_{ta,\d+}_{length,\d+}_Ne{Ne,\d+}_{rep,\d+}.scale.npz'
  run:
    # loading in the data
    cur_data = np.load(input.hap_panel)
    hap_panel_test = cur_data['haps']
    pos = cur_data['rec_pos']
    # Extracting the panel
    modern_hap_panel = hap_panel_test[:-1,:]
    # Test haplotype in simulations is always the last haplotype...
    test_hap = hap_panel_test[-1,:]
    # Setting this up properly...
    cur_hmm = LiStephensHMM(haps = modern_hap_panel, positions=pos)
    # Setting theta here...
    cur_hmm.theta = cur_hmm._infer_theta()
    # Setting the error rate to be similar to the original LS-Model
    eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
    scales = np.logspace(2,6,50)
    neg_log_lls = np.array([cur_hmm._negative_logll(test_hap, scale=s, eps=eps) for s in scales])
    mle_scale = cur_hmm._infer_scale(test_hap, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
    # Estimating both error and scale parameters jointly
    mle_params = cur_hmm._infer_params(test_hap, x0=[1e2,1e-3], bounds=[(1.,1e6),(1e-6,0.9)], tol=1e-7)
    cur_params = np.array([np.nan,np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    model_params = np.array([modern_hap_panel.shape[0], pos.size])
    np.savez(output.log_ll, scales=scales, loglls=-neg_log_lls, scale=mle_scale['x'], params=cur_params, model_params=model_params)    


rule infer_scale_serial_test_ascertained:
  """ 
    Infer scale parameter using a naive Li-Stephens Model 
      and ascertaining to snps in the modern panel at a high-frequency
    NOTE : should we have considerations for 
  """
  input:
    hap_panel = rules.create_hap_panel_test.output.hap_panel
  wildcard_constraints:
    scenario = '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialMigration_[0-9]*|SerialBottleneck_[0-9]*_[0-9]*_[0-9]*)'
  output:
    log_ll = 'data/hap_copying/mle_results/{scenario}/mle_scale_{mod_n, \d+}_{n_anc, \d+}_{ta,\d+}_{length,\d+}_Ne{Ne,\d+}_{rep,\d+}.{asc,\d+}.scale.npz'
  run:
    # loading in the data
    cur_data = np.load(input.hap_panel)
    hap_panel_test = cur_data['haps']
    pos = cur_data['rec_pos']
    times = cur_data['ta']
    times = times.astype(np.float32)
    # Testing things out here ... 
    mod_idx = np.where(times == 0)[0]
    ta_test = np.int32(wildcards.ta)
    ta_idx = np.where(times == ta_test)[0][0]
    # Extracting the panel
    modern_hap_panel = hap_panel_test[mod_idx,:]
    # Test haplotype in simulations is always the last haplotype...
    test_hap = hap_panel_test[ta_idx,:]
    mod_asc_panel, asc_pos, asc_idx = ascertain_variants(modern_hap_panel, pos, maf = np.int32(wildcards.asc)/100.)
    anc_asc_hap = test_hap[asc_idx]
    
    afreq_mod = np.sum(mod_asc_panel, axis=1)
    
    cur_hmm = LiStephensHMM(haps = mod_asc_panel, positions=asc_pos)
    # Setting theta here ...
    cur_hmm.theta = cur_hmm._infer_theta()
    # Setting the error rate to be similar to the original LS-Model
    eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
    scales = np.logspace(2,6,25)
    neg_log_lls = np.array([cur_hmm._negative_logll(anc_asc_hap, scale=s, eps=eps) for s in tqdm(scales)])
    mle_scale = cur_hmm._infer_scale(anc_asc_hap, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
    # Estimating both error and scale parameters jointly
    mle_params = cur_hmm._infer_params(anc_asc_hap, x0=[1e2,1e-3], bounds=[(1.,1e7), (1e-6,0.9)], tol=1e-7)
    cur_params = np.array([np.nan, np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
    np.savez(output.log_ll, scales=scales, loglls=-neg_log_lls, scale=mle_scale['x'], params=cur_params, model_params=model_params, mod_freq = afreq_mod)        
    
    
rule gen_all_hap_panels:
  input:
    expand('data/hap_copying/hap_panels/{scenario}/hap_panel_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{rep}.npz', mod_n=[100], n_anc=1, ta=[0], scenario=['SerialConstant'], length=20, rep=np.arange(10), Ne=Ne),
    expand('data/full_sim_all/{scenario}/generations_{ta}_{interval}/hap_panel_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.panel.npz', scenario='SerialConstant', ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=10000, rep=np.arange(5)),
    expand('data/full_sim_all/{scenario}/generations_{ta}_{interval}/hap_panel_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.panel.npz', scenario='IBDNeUK10K', ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=10000, rep=np.arange(5))

    
rule calc_infer_scales_all:
  """ Test rule to infer parameters under the LS-Model """
  input: 
    expand('data/hap_copying/mle_results/{scenario}/mle_scale_{mod_n}_{n_anc}_{ta}_{length}_Ne{Ne}_{rep}.scale.npz', mod_n=[100], n_anc=1, ta=np.arange(0,2001,100), scenario=['SerialConstant', 'SerialBottleneck', 'SerialBottleneckLate','TennessenEuropean'], length=20, rep=np.arange(10), Ne=Ne),
    expand('data/hap_copying/mle_results/{scenario}/mle_scale_{mod_n}_{n_anc}_{ta}_{length}_Ne{Ne}_{rep}.scale.npz', mod_n=100, n_anc=1, ta=np.arange(0,1001,100), scenario= ['SerialMigration_4','SerialMigration_3', 'SerialMigration_2', 'SerialMigration_1'], length=20, rep=np.arange(10), Ne=Ne),


    
    
    
    
    
    
    
    
rule calc_infer_scales_asc_all_figures:
  """Rule to calculate all of the simulations for figures on haplotype copying models as a function of time
  """
  input:
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialConstant'], ta=1000, interval=10, mod_n=100, n_anc=1, length=40, Ne=[100000,20000, 10000,5000], rep=np.arange(5), asc=5, ta_samp=np.arange(50,1001,50)),
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['IBDNeUK10K', 'TennessenEuropean'], ta=1000, interval=10, mod_n=100, n_anc=1, length=40, Ne=[10000], rep=np.arange(5), asc=5, ta_samp=np.arange(50,1001,50)),
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialConstant', 'SerialBottleneckInstant1', 'SerialBottleneckInstant2', 'SerialBottleneckInstant3'], ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=10000, rep=np.arange(5), asc=5, ta_samp=np.arange(5,401,5)),
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SerialBottleneckInstant7', 'SerialBottleneckInstant8', 'SerialBottleneckInstant9'], ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=1000000, rep=np.arange(5), asc=5, ta_samp=np.arange(5,401,5)),
    expand('data/hap_copying/mle_results_all/{scenario}/generations_{ta}_{interval}/mle_scale_{mod_n}_{n_anc}_{length}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz', scenario=['SimpleGrowth1', 'SimpleGrowth2', 'SimpleGrowth3', 'SimpleGrowth4'], ta=400, interval=5, mod_n=100, n_anc=1, length=40, Ne=1000000, rep=np.arange(5), asc=5, ta_samp=np.arange(5,401,5))

    
    
    
rule calc_infer_scales_all_test_sample_size:
  input:
    expand('data/hap_copying/mle_results/{scenario}/mle_scale_{mod_n}_{n_anc}_{ta}_{length}_Ne{Ne}_{rep}.scale.npz', mod_n=[100,200,500], n_anc=1, ta=[10], scenario='SerialConstant', length=20, rep=np.arange(5), Ne=Ne)
    

#     expand('data/hap_copying/changepts/{scenario}/changepts_{mod_n}_{n_anc}_{ta}_{length}_Ne{Ne}_totals.npy', mod_n=[100], n_anc=1, ta=np.arange(0,2001,100), scenario='SerialConstant', Ne=Ne, length=20),
# rule check_corr_seg_sites:
#   """
#     Estimate the correlation in segregating sites in different demographic scenarios
#   """
#   input:
#     expand('data/hap_copying/hap_panels/{scenario}/hap_panel_{mod_n}_{n_anc}_{ta}_{length}_Ne_10000_{rep}.npz', mod_n=1, n_anc=1, rep=np.arange(20), scenario=['SerialConstant', 'TennessenEuropean'], ta=[0,100,1000, 10000], length=20)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# -------- Error Checking ----------- # 
rule infer_scale_missing_data:
  """
    Test effect of missing data on scale inference ... 
  """
  input:
    hap_panel = rules.create_hap_panel_test.output.hap_panel
  wildcard_constraints:
    scenario = '(SerialConstant|SerialBottleneck|SerialBottleneckLate|TennessenEuropean|SerialMigration_[0-9]*)'
  output:
    log_ll = 'data/hap_copying/mle_results/{scenario}/missing_test/mle_scale_{mod_n, \d+}_{n_anc, \d+}_{ta,\d+}_{length,\d+}_Ne{Ne,\d+}_{rep,\d+}.miss_{miss,\d+}_{repmiss,\d+}.scale.npz'
  run:
    # loading in the data
    cur_data = np.load(input.hap_panel)
    hap_panel_test = cur_data['haps']
    pos = cur_data['rec_pos']
    modern_hap_panel = hap_panel_test[:-1,:]
    test_hap = hap_panel_test[-1,:]
    # Randomly setting some proportion missing
    miss_prop = int(wildcards.miss) / 100
    assert((miss_prop >= 0.) & (miss_prop < 1.0))
    n_indiv, n_snps = modern_hap_panel.shape
    miss_idx = np.random.choice(n_snps, int((1. - miss_prop) * n_snps), replace=False)
    miss_idx = np.sort(miss_idx)
    modern_hap_panel_miss = modern_hap_panel[:,miss_idx]
    pos_miss = pos[miss_idx]
    test_hap_miss = test_hap[miss_idx]
    print('Established Missing SNPs')
    cur_hmm = LiStephensHMM(haps = modern_hap_panel_miss, positions=pos_miss)
    # Setting theta here...
    cur_hmm.theta = cur_hmm._infer_theta()
    # Setting the error rate to be similar to the original LS-Model
    eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
    print(eps)
    scales = np.logspace(2, 6, 50)
    neg_log_lls = np.array([cur_hmm._negative_logll(test_hap_miss, scale=s, eps=eps) for s in tqdm(scales)])
    mle_scale = cur_hmm._infer_scale(test_hap_miss, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
    mle_params = cur_hmm._infer_params(test_hap_miss, x0=[1e2, 1e-3], bounds=[(1e1,1e7),(1e-6,0.9)], tol=1e-7)
    cur_params = np.array([np.nan,np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    model_params = np.array([n_indiv, n_snps, miss_prop, int(wildcards.ta)])
    np.savez(output.log_ll, scales=scales, loglls=-neg_log_lls, scale=mle_scale['x'], mle_params=cur_params, model_params=model_params) 
    
    
rule test_missing_scale_inference:
  input:
    expand('data/hap_copying/mle_results/{scenario}/missing_test/mle_scale_{mod_n}_{n_anc}_{ta}_{length}_Ne{Ne}_{rep}.miss_{miss}_{repmiss}.scale.npz', scenario='SerialConstant', mod_n=100, n_anc=1, ta=np.arange(0,1001,100), length=20, Ne=10000, rep=[0], miss=[0,5,25,50], repmiss=np.arange(5))

# ----------  Simulate Chromosome w. Realistic Recombination Map   ----------- # 
rule sim_format_recombination_map:
  input:
    recmap = 'data/maps_b37/maps_chr.{CHROM}'
  output:
    recmap_hapmap_format = 'data/hap_copying/real_chrom_sims/chr{CHROM}_{genmap}.map.txt',
  wildcard_constraints:
    genmap = '(deCODE|COMBINED_LD)',
    CHROM = '0-9*|X'
  run:
    total_df = pd.read_csv(input.recmap, sep='\s+')
    filtered_map  = total_df[['Physical_Pos', wildcards.genmap]]
    nsnps , _ = filtered_map.shape
    cm_diff = filtered_map[wildcards.genmap].values[1:] - filtered_map[wildcards.genmap].values[:-1]
    pos_diff_Mb = (filtered_map['Physical_Pos'].values[1:] - filtered_map['Physical_Pos'].values[:-1]) / 1e6
    cm_per_Mb = cm_diff / pos_diff_Mb
    cm_per_Mb = np.insert(cm_per_Mb, 0, 0.0)
    filtered_map['cM/Mb'] = cm_per_Mb
    filtered_map['Chromosome'] = np.repeat(wildcards.CHROM, nsnps)
    filtered_map = filtered_map[['Chromosome','Physical_Pos','cM/Mb', wildcards.genmap]]
    filtered_map.to_csv(output.recmap_hapmap_format, sep=' ', index=False)

    
rule gen_modern_hap_panel_real_map:
  input:
    recmap = rules.sim_format_recombination_map.output.recmap_hapmap_format
  output:
    hap_panel = 'data/hap_copying/real_chrom_sims/panels/chr{CHROM,\d+}_{genmap}.n{n,\d+}.rep{rep,\d+}.panel.npz'
  run:
    df_recmap = pd.read_csv(input.recmap, sep='\s+')
    recmap = msp.RecombinationMap.read_hapmap(input.recmap)
    ts = msp.simulate(sample_size=int(wildcards.n), recombination_map=recmap, mutation_rate=1.2e-8, Ne=1e4)
    pos = np.array([s.position for s in ts.sites()])
    haps = ts.genotype_matrix()
    daf = np.mean(haps, axis=1)
    idx = (daf > 0.01) & (daf < 0.99)
    haps_filt = haps[idx,:]
    pos_filt = pos[idx]
    cm_pos_filt = np.interp(pos_filt, df_recmap['Physical_Pos'].values, df_recmap[wildcards.genmap].values)
    # Converting it to morgan positioning
    morgan_pos_filt = cm_pos_filt / 100
    pos_diff = morgan_pos_filt[1:] - morgan_pos_filt[:-1]
    idx_diff = pos_diff > 0.
    idx_diff = np.insert(idx_diff,True,0)
    # Second filtering step for positions within the recombination map
    haps_filt = haps_filt[idx_diff,:]
    morgan_pos_filt = morgan_pos_filt[idx_diff]
    pos_filt = pos_filt[idx_diff]
    # TODO : do you just filter all positions that have zero recombinational distance between them?
    np.savez(output.hap_panel, haps=haps_filt.T, rec_pos=morgan_pos_filt, phys_pos=pos_filt)
    
rule infer_scale_simmed_real_map:
  input:
    hap_panel = rules.gen_modern_hap_panel_real_map.output.hap_panel
  output:
    scale_inf = 'data/hap_copying/real_chrom_sims/results/chr{CHROM,\d+}_{genmap}.n{n}.scale_{scale_min,\d+}_{scale_max,\d+}.seed{seed,\d+}.rep{rep,\d+}.npz'
  run:
    scale_min, scale_max = int(wildcards.scale_min), int(wildcards.scale_max)
    assert(scale_min < scale_max)
    assert(scale_min % 100 == 0)
    assert(scale_max % 100 == 0)
    scales_true = np.arange(scale_min, scale_max, 100)
    df = np.load(input.hap_panel)
    ls_model = LiStephensHMM(df['haps'], df['rec_pos'])
    test_haps = [ls_model._sim_haplotype(scale=s, eps=1e-2, seed=int(wildcards.seed))[0] for s in scales_true]
    scales_hat = np.zeros(scales_true.size)
    for i in tqdm(range(scales_true.size)):
      res = ls_model._infer_scale(test_haps[i], eps=1e-2, method='Bounded', bounds=[1e1,1e5], tol=1e-3)
      inf_scale = res['x']
      scales_hat[i] = inf_scale
    # Saving the output file
    np.savez(output.scale_inf, true_scales=scales_true, scales_hat=scales_hat)
    
rule infer_scales_real_genmap:
  input:
    expand('data/hap_copying/real_chrom_sims/results/chr{CHROM}_{genmap}.n{n}.scale_{scale_min}_{scale_max}.seed{seed}.rep{rep}.npz', seed=np.arange(1,11), rep=0, genmap='deCODE', CHROM=22, scale_max=2500, scale_min=100, n=[100])


rule infer_scale_simmed_uniform_map:
  input:
    hap_panel = rules.create_hap_panel_test.output.hap_panel
  output:
    scale_inf = 'data/hap_copying/full_sim_results_unif/{scenario}/hap_panel_{mod_n, \d+}_{n_anc, \d+}_{ta, \d+}_{length, \d+}_Ne_{Ne,\d+}_{rep,\d+}.scale_{scale_min,\d+}_{scale_max,\d+}.seed{seed,\d+}.rep{rep}.npz'
  run:
    scale_min, scale_max = int(wildcards.scale_min), int(wildcards.scale_max)
    assert(scale_min < scale_max)
    assert(scale_min % 100 == 0)
    assert(scale_max % 100 == 0)
    scales_true = np.arange(scale_min, scale_max, 100)
    df = np.load(input.hap_panel)
    ls_model = LiStephensHMM(df['haps'], df['rec_pos'])
    test_haps = [ls_model._sim_haplotype(scale=s, eps=1e-2, seed=int(wildcards.seed))[0] for s in scales_true]
    scales_hat = np.zeros(scales_true.size)
    for i in tqdm(range(scales_true.size)):
      res = ls_model._infer_scale(test_haps[i], eps=1e-2, method='Bounded', bounds=[1e1,1e5], tol=1e-3)
      inf_scale = res['x']
      scales_hat[i] = inf_scale
    # Saving the output file
    np.savez(output.scale_inf, true_scales=scales_true, scales_hat=scales_hat)

    
rule infer_scales_full_unif_map:
  input:
    expand('data/hap_copying/full_sim_results_unif/{scenario}/hap_panel_{mod_n}_{n_anc}_{ta}_{length}_Ne_{Ne}_{rep}.scale_{scale_min}_{scale_max}.seed{seed}.rep{rep}.npz', seed=np.arange(1,11), rep=0, genmap='deCODE', length=20, Ne=10000, scenario='SerialConstant', mod_n=[100], n_anc=[1], ta=[0], scale_max=2000, scale_min=100)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ---------- Running a serial simulation according to the real Reich Lab Data  -----------# 

gen_time = 30

# TODO : should we simulate a specific chromosome?
rule sim_serial_data_1kg_ceu_real_chrom:
  """
    Simulating example data for the CEU setting  
  """
  input:
    recmap = rules.sim_format_recombination_map.output.recmap_hapmap_format,
    times_ceu_km = lambda wildcards: 'data/hap_copying/time_points/ceu_%dkm_ages.npy' % int(wildcards.km),
    
  output:
    treeseq = 'data/full_sim_all/{scenario}/ceu_sim_{km, \d+}_{CHROM}_{genmap}/serial_coal_{mod_n, \d+}_Ne{Ne,\d+}_{rep,\d+}.treeseq.gz'
  wildcard_constraints:
    scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)'
  run:
    times_kya = np.load(input.times_ceu_km)
    times_gen = np.array(times_kya / gen_time, dtype=np.int32)
    unique_times, n_samples = np.unique(times_gen, return_counts=True)
    recmap = msp.RecombinationMap.read_hapmap(input.recmap)
    Ne = np.int32(wildcards.Ne)
    mod_n = np.int32(wildcards.mod_n)
    t_anc = unique_times.tolist()
    n_anc = n_samples.tolist()
    scenario = wildcards.scenario
    cur_sim = None
    if scenario == 'SerialConstant':
      cur_sim = SerialConstant(Ne=Ne, mod_n=mod_n, t_anc=t_anc, n_anc=n_anc)
    elif scenario == 'TennessenEuropean':
      cur_sim = SerialTennessenModel()
      cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
    elif scenario == 'TennessenDoubleGrowthEuropean':
      cur_sim = SerialTennessenModel()
      cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
      cur_sim._change_growth_rate(r_EU=0.039)
    elif scenario == 'TennessenQuadGrowthEuropean':
      cur_sim = SerialTennessenModel()
      cur_sim._add_samples(mod_pop=1, anc_pop=1, n_mod=mod_n, n_anc=n_anc, t_anc=t_anc)
      cur_sim._change_growth_rate(r_EU=0.078)
    elif scenario == 'IBDNeUK10K':
      cur_sim = SerialIBDNeUK10K(demo_file=ibd_ne_demo_file)
      cur_sim._set_demography()
      cur_sim._add_samples(n_mod = mod_n, n_anc = n_anc, t_anc = t_anc)
    else:
      raise ValueError('Improper scenario input for this simulation!')
    # Conducting the actual simulation ... 
    tree_seq = cur_sim._simulate(mutation_rate=mut_rate, recombination_map=recmap)
    tszip.compress(tree_seq, str(output.treeseq))

rule create_hap_panel_1kg_ceu_real_chrom:
  """
    Creates a haplotype panel for a joint simulation 
  """
  input:
    treeseq = rules.sim_serial_data_1kg_ceu_real_chrom.output.treeseq
  wildcard_constraints:
    scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)',
    genmap = '(deCODE|COMBINED_LD)'
  output:
    hap_panel = 'data/full_sim_all/{scenario}/ceu_sim_{km, \d+}_{CHROM}_{genmap}/serial_coal_{mod_n, \d+}_Ne{Ne,\d+}_{rep,\d+}.panel.npz'
  run:
    ts = tszip.decompress(input.treeseq)
    geno = ts.genotype_matrix().T
    phys_pos = np.array([v.position for v in ts.variants()])
    rec_pos = phys_pos * rec_rate
    node_ids = [s for s in ts.samples()]
    tree = ts.first()
    times = np.array([tree.time(x) for x in node_ids])
    np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times)    
    

rule infer_scale_serial_ascertained_ceu_sims:
  """ 
    Infer scale parameter using a naive Li-Stephens Model 
      and ascertaining to snps in the modern panel at 
  """
  input:
    hap_panel = rules.create_hap_panel_1kg_ceu_real_chrom.output.hap_panel,
    times_ceu_km = lambda wildcards: 'data/hap_copying/time_points/ceu_%dkm_ages.npy' % int(wildcards.km)
  wildcard_constraints:
    scenario = '(SerialConstant|TennessenEuropean|TennessenDoubleGrowthEuropean|TennessenQuadGrowthEuropean|IBDNeUK10K)',
    genmap = '(deCODE|COMBINED_LD)'
  output:
    mle_hap_est = 'data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km,\d+}_{CHROM}_{genmap}/mle_scale_{mod_n, \d+}_Ne{Ne,\d+}_{rep, \d+}.asc_{asc, \d+}.ta_{ta_samp, \d+}.scale.npz'
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
    # Setting theta here ...
    cur_hmm.theta = cur_hmm._infer_theta()
    # Setting the error rate to be similar to the original LS-Model
    eps = cur_hmm.theta/(cur_hmm.n_samples + cur_hmm.theta)
    mle_scale = cur_hmm._infer_scale(anc_asc_hap, eps=eps, method='Bounded', bounds=(1.,1e6), tol=1e-7)
    # Estimating both error and scale parameters jointly
    mle_params = cur_hmm._infer_params(anc_asc_hap, x0=[1e2,1e-3], bounds=[(1.,1e7), (1e-6,0.9)], tol=1e-7)
    cur_params = np.array([np.nan, np.nan])
    if mle_params['success']:
      cur_params = mle_params['x']
    model_params = np.array([mod_asc_panel.shape[0], asc_pos.size, ta_test])
    np.savez(output.mle_hap_est, scale=mle_scale['x'], params=cur_params, model_params=model_params, mod_freq = afreq_mod)


# # Actually loading in some of these schemes 
# test_1500_km = np.load('data/hap_copying/time_points/ceu_1500km_ages.npy')/gen_time
# test_1500_km = np.unique(test_1500_km.astype(int))

# test_2500_km = np.load('data/hap_copying/time_points/ceu_2500km_ages.npy')/gen_time
# test_2500_km = np.unique(test_2500_km.astype(int))

# rule test_ceu_infer_scale_real_chrom:
#   input:
# #     expand('data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km}_{CHROM}_deCODE/mle_scale_{mod_n}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz',scenario=['SerialConstant', 'TennessenEuropean', 'TennessenDoubleGrowthEuropean','TennessenQuadGrowthEuropean'], km=2500, mod_n=49, CHROM='X', Ne=10000, rep=0, asc=10, ta_samp=test_2500_km),
#     expand('data/hap_copying/mle_results_all/{scenario}/ceu_sim_{km}_{CHROM}_deCODE/mle_scale_{mod_n}_Ne{Ne}_{rep}.asc_{asc}.ta_{ta_samp}.scale.npz',scenario=['SerialConstant', 'TennessenEuropean', 'IBDNeUK10K'], km=1500, mod_n=49, CHROM='X', Ne=10000, rep=0, asc=5, ta_samp=test_1500_km)    

    
