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
from time_strat_ld import TimeStratifiedLDStats

# Relying on some of the
include: "hap_copying.smk"

# Import configurations to add some things
configfile: "config.yml"

### --- Simulation Parameters --- ###
lengths = [5,10,20]
rec_rate = 1e-8
mut_rate = 1e-8
Ne = 10000
nreps=20
seeds = [i+1 for i in range(nreps)]

###### --------- Estimating LD-statistics from haplotype copying simulations ----------- ######

rule est_LDjt_stats_raw:
  input:
    hap_panel = rules.sim_demography_single_ancients.output.hap_panel
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
    pABmod, pABanc,pAmod, pAanc, pBmod, pBanc, Dmod, Danc, gen_dist = TimeStratifiedLDStats.time_strat_hap_freq(haps, rec_pos, times, ta=ta, maf=maf, m=10000, polymorphic_total=polytot, seed=seed)
    # Save the function here ...
    np.savez_compressed(output.ld_stats, pABmod=pABmod, pABanc=pABanc, pAmod=pAmod, pBmod=pBmod, pAanc=pAanc, pBanc=pBanc, Dmod=Dmod, Danc=Danc, rec_dist=gen_dist, ta=ta, scenario=scenario, seed=seed)



rule est_LDjt_stats:
  input:
    hap_panel = rules.sim_demography_single_ancients.output.hap_panel
  output:
    ld_stats = config['tmpdir'] + 'ld_stats/{scenario}/jointLDstats_mod{mod_n,\d+}_anc{n_anc}_t{ta,\d+}_l{length,\d+}_Ne{Ne,\d+}_{seed,\d+}_maf{maf,\d+}_polytotal{polytot,\d+}.npz'
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
    ed0dt_norm, rec_dist, mod_af, anc_af = TimeStratifiedLDStats.joint_ld(haps, rec_pos, times, ta=ta, maf=maf, polymorphic_total=polytot)
    # Save the function here ...
    np.savez_compressed(output.ld_stats, ed0dt=ed0dt_norm, rec_dist=rec_dist, mod_af=mod_af, anc_af=anc_af, ta=ta, scenario=scenario, seed=seed)

rule run_est_jtLDstats:
  input:
#     expand(config['tmpdir'] + 'ld_stats/{scenario}/jointLDstats_mod{mod_n}_anc{n_anc}_t{ta}_l{length}_Ne{Ne}_{seed}_maf{maf}_polytotal{polytot}.npz', scenario='SerialConstant', ta=100, mod_n=500, n_anc=500, seed=seeds[0], length=1, Ne=10000, maf=[1,5], polytot=[0,1]),
#     expand(config['tmpdir'] + 'ld_stats_raw/{scenario}/jointLDstats_mod{mod_n}_anc{n_anc}_t{ta}_l{length}_Ne{Ne}_{seed}_maf{maf}_polytotal{polytot}.npz', scenario='SerialConstant', ta=100, mod_n=100, n_anc=100, seed=seeds[0], length=[1], Ne=10000, maf=[1,5,10], polytot=[0,1]),
#     expand(config['tmpdir'] + 'ld_stats_raw/{scenario}/jointLDstats_mod{mod_n}_anc{n_anc}_t{ta}_l{length}_Ne{Ne}_{seed}_maf{maf}_polytotal{polytot}.npz', scenario='SerialConstant', ta=100, mod_n=500, n_anc=500, seed=seeds, length=[1], Ne=10000, maf=[5], polytot=[0,1]),
    expand(config['tmpdir'] + 'ld_stats_raw/{scenario}/jointLDstats_mod{mod_n}_anc{n_anc}_t{ta}_l{length}_Ne{Ne}_{seed}_maf{maf}_polytotal{polytot}.npz', scenario='SerialConstant', ta=[0,100,1000], mod_n=500, n_anc=500, seed=seeds[:10], length=[1], Ne=10000, maf=[5], polytot=[0,1])
