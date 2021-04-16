#!python3

import os
import sys
import numpy as np
import msprime as msp
import tszip
import allel
import pandas as pd
from tqdm import tqdm
from scipy.stats import binned_statistic

sys.path.append('src/')
from time_strat_ld import TimeStratifiedLDStats

# Import configurations to add some things
configfile: "config.yml"

### --- Simulation Parameters --- ###
lengths = [5,10,20]
rec_rate = 2e-8
mut_rate = 5e-8
Ne = 10000
nreps=20
seeds = [i+1 for i in range(nreps)]

###### --------- Estimating LD-statistics from haplotype simulations ----------- ######

rule sim_demography_single_ancients_ldstats:
  """
    Generate a tree-sequence of multiple ages
  """
  output:
    treeseq= config['tmpdir'] + 'hap_copying_raw/hap_panels/{scenario}/hap_panel_{mod_n,\d+}_{n_anc}_{ta,\d+}_{length,\d+}_Ne_{Ne,\d+}_{seed,\d+}.treeseq.gz',
    hap_panel=config['tmpdir'] + 'hap_copying_raw/hap_panels/{scenario}/hap_panel_{mod_n,\d+}_{n_anc}_{ta,\d+}_{length,\d+}_Ne_{Ne,\d+}_{seed,\d+}.npz',
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
    seed = np.int32(wildcards.seed)
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
    ts = cur_sim._simulate(mutation_rate=mut_rate, recombination_rate=rec_rate, length=length, random_seed=seed)
    tszip.compress(ts, str(output.treeseq))
    # Generating the haplotype reference panel...
    geno = ts.genotype_matrix().T
    phys_pos = np.array([v.position for v in ts.variants()])
    rec_pos = phys_pos * rec_rate
    node_ids = [s for s in ts.samples()]
    tree = ts.first()
    times = np.array([tree.time(x) for x in node_ids])
    np.savez_compressed(output.hap_panel, haps=geno, rec_pos=rec_pos, phys_pos=phys_pos, ta=times, scenario=scenario, seed=seed)

    
rule est_LDjt_stats_raw:
  """Estimate the raw haplotype frequencies."""
  input:
    hap_panel = rules.sim_demography_single_ancients_ldstats.output.hap_panel
  output:
    ld_stats = config['tmpdir'] + 'ld_stats_raw/{scenario}/jointLDstats_mod{mod_n,\d+}_anc{n_anc}_t{ta,\d+}_l{length,\d+}_Ne{Ne,\d+}_{seed,\d+}_maf{maf,\d+}_polytotal{polytot,\d+}.npz'
  run:
    haps_df = np.load(input.hap_panel)
    maf = int(wildcards.maf) / 100.
    haps = haps_df['haps']
    polytot = bool(wildcards.polytot)
    times = haps_df['ta']
    rec_pos = haps_df['rec_pos']
    phys_pos = haps_df['phys_pos']
    scenario = haps_df['scenario']
    seed = haps_df['seed']
    ta = int(wildcards.ta)
    # Actually compute the joint product of LD
    pABmod, pABanc,pAmod, pAanc, pBmod, pBanc, Dmod, Danc, gen_dist = TimeStratifiedLDStats.time_strat_hap_freq(haps, rec_pos, times, ta=ta, maf=maf, m=10000, dist_bins='auto', polymorphic_total=polytot, seed=seed)
    # Save the function here ...
    np.savez_compressed(output.ld_stats, pABmod=pABmod, pABanc=pABanc, pAmod=pAmod, pBmod=pBmod, pAanc=pAanc, pBanc=pBanc, Dmod=Dmod, Danc=Danc, rec_dist=gen_dist, ta=ta, scenario=scenario, seed=seed)


# rule est_LDjt_stats:
#   input:
#     hap_panel = rules.sim_demography_single_ancients_ldstats.output.hap_panel
#   output:
#     ld_stats = config['tmpdir'] + 'ld_stats/{scenario}/jointLDstats_mod{mod_n,\d+}_anc{n_anc}_t{ta,\d+}_l{length,\d+}_Ne{Ne,\d+}_{seed,\d+}_maf{maf,\d+}_polytotal{polytot,\d+}.npz'
#   run:
#     haps_df = np.load(input.hap_panel)
#     maf = int(wildcards.maf) / 100.
#     haps = haps_df['haps']
#     polytot = bool(wildcards.polytot)
#     times = haps_df['ta']
#     rec_pos = haps_df['rec_pos']
#     phys_pos = haps_df['phys_pos']
#     scenario = haps_df['scenario']
#     seed = haps_df['seed']
#     ta = int(wildcards.ta)
#     # Actually compute the joint product of LD
#     ed0dt_norm, rec_dist, mod_af, anc_af = TimeStratifiedLDStats.joint_ld(haps, rec_pos, times, ta=ta, maf=maf, polymorphic_total=polytot)
#     # Save the function here ...
#     np.savez_compressed(output.ld_stats, ed0dt=ed0dt_norm, rec_dist=rec_dist, mod_af=mod_af, anc_af=anc_af, ta=ta, scenario=scenario, seed=seed)

    

rule run_est_jtLDstats_raw:
  input:
    expand(config['tmpdir'] +
            'ld_stats_raw/{scenario}/jointLDstats_mod{mod_n}_anc{n_anc}_t{ta}_l{length}_Ne{Ne}_{seed}_maf{maf}_polytotal{polytot}.npz',
            scenario='SerialConstant', ta=[0,100,1000], mod_n=500, n_anc=500,
            seed=seeds[10:], length=1, Ne=10000, maf=[5], polytot=[1])
  
# ------ Collapsing these estimates into a plottable format ----- #
rule collapse_raw_stats:
  input:
    ldfiles = rules.run_est_jtLDstats_raw.input
  output:
     results = 'results/ld_stats_raw/ld_stats_time_sep_raw.csv.gz'
  run:
    ed0dt_tot = []
    std_ed0dt_tot = []
    rec_dist_tot = []
    ta_tot = []
    scenario_tot = []
    seed_tot = []
    for f in tqdm(input.ldfiles):
      df = np.load(f)
      # Calculate the expected LD statistic 
      ed0dt = eD0Dt_norm = (df['Dmod']*df['Danc']) / (df['pAmod']*(1.-df['pAanc'])*df['pBmod']*(1. - df['pBanc']))
      rec_dist = df['rec_dist']
      _, bins = np.histogram(rec_dist, bins='auto')
      mean_ed0dt, _, _ = binned_statistic(rec_dist, ed0dt, np.nanmean, bins=bins)
      std_ed0dt, _, _ = binned_statistic(rec_dist, ed0dt, np.nanstd, bins=bins)
      mean_bins = (bins[1:] + bins[:-1])/2.
      print(mean_ed0dt.size, std_ed0dt.size, mean_bins.size)
      assert mean_ed0dt.size == mean_bins.size
      ta = np.repeat(df['ta'], mean_bins.size)
      scenario = np.repeat(df['scenario'], mean_bins.size)
      seed = np.repeat(df['seed'], mean_bins.size)
      # Append them all!
      ed0dt_tot.append(mean_ed0dt)
      std_ed0dt_tot.append(std_ed0dt)
      rec_dist_tot.append(mean_bins)
      ta_tot.append(ta)
      scenario_tot.append(scenario)
      seed_tot.append(seed)
    ed0dt_tot = np.hstack(ed0dt_tot)
    std_ed0dt_tot = np.hstack(std_ed0dt_tot)
    rec_dist_tot = np.hstack(rec_dist_tot)
    ta_tot = np.hstack(ta_tot)
    scenario_tot = np.hstack(scenario_tot)
    seed_tot = np.hstack(seed_tot)
    
    tot_dict = {'ed0dt': ed0dt_tot, 'std_ed0dt': std_ed0dt_tot, 'rec_dist': rec_dist_tot, 'ta': ta_tot, 'scenario': scenario_tot, 'seed': seed_tot}
    df_out = pd.DataFrame(tot_dict)
    df_out.to_csv(output.results, index=False)
    