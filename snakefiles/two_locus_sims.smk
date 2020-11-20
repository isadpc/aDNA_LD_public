"""Running two-locus simulations to check Monte-Carlo results."""

import os
import sys
import numpy as np
import msprime as msp
import tszip
import pandas as pd
from tqdm import tqdm

# Import all of the functionality that we would need
sys.path.append('src/')
from coal_cov import *

# Import configurations to add some things
configfile: "config.yml"

# Data file for IBDNe UK10K demography
ibd_ne_demo_file = 'data/demo_models/uk10k.IBDNe.txt'

## Approximating different sequence differences
# rho = 400 -> 1 Mb
# rho = 40 -> 100 kb
# rho = 4 -> 10 kb
rhos2 = [0.01,0.1,1,10,100,500,1000]

# Simulate
tas2 = np.array([0.0, 0.001, 0.01, 0.1, 1.0])

# # Simulate at a number of divergence times
# tdivs = [0.001, 0.01, 0.1]

# Global number of replicates to simulate per parameter combination?
nreps = 100000

#### ---------- 1. Serial Coalescent Model --------------- ####
"""
  Simulate Two-Loci TMRCA w/ serial sampling
"""
rule sim_two_locus_tmrcas:
  output:
    config['tmpdir'] + 'two_loci/serial/tmrca_{n0,\d+}_{na,\d+}/{rho}_{ta}_{nreps,\d+}_tmrca.npy'
  run:
    Ne=1e4
    rho = np.float32(wildcards.rho)
    rec_rate = rho / (4.*Ne)
    ta = np.float32(wildcards.ta)
    nreps = np.int32(wildcards.nreps)
    na = np.int32(wildcards.na)
    n0 = np.int32(wildcards.n0)
    cur_two_locus = TwoLocusSerialCoalescent(ta=ta*2.*Ne, Ne=Ne, rec_rate=rec_rate, na=na, n0=n0, reps=nreps)
    cur_two_locus._simulate()
    cur_two_locus._two_locus_tmrca()
    np.save(str(output), cur_two_locus.pair_tmrca)

"""
  Simulate two-locus branch lengths (useful for calculating empirical covariances!)
"""
rule sim_two_locus_branch_length:
  output:
    config['tmpdir'] + 'two_loci/serial/branch_length_{n0, \d+}_{na,\d+}/{rho}_{ta}_{nreps,\d+}_branch_length.npy'
  run:
    Ne=1e4
    rho = np.float32(wildcards.rho)
    rec_rate = rho / (4.*Ne)
    ta = np.float32(wildcards.ta)
    nreps = np.int32(wildcards.nreps)
    na = np.int32(wildcards.na)
    n0 = np.int32(wildcards.n0)
    cur_two_locus = TwoLocusSerialCoalescent(ta=ta*2.*Ne, Ne=Ne, rec_rate=rec_rate, na=na, n0=n0, reps=nreps)
    cur_two_locus._simulate()
    cur_two_locus._two_locus_branch_length()
    np.save(str(output), cur_two_locus.pair_branch_length)


# ------- Finalized simulations under different demographic histories -------#
rule run_two_locus_sims_scenarios:
  output:
    config['tmpdir'] + 'two_loci/demographies/{scenario}/two_locus_sims_n0{n0,\d+}_na{na,\d+}.ta{ta,\d+}.r_{rec_rate, \d+}.Ne{Ne,\d+}.rep{nreps,\d+}.branch_length.npz'
  wildcard_constraints:
    scenario='SerialConstant|IBDNeUK10K|Tennessen|InstantGrowth[0-9]*'
  run:
    rec_rate = np.float32(wildcards.rec_rate)
    rec_rate = 10**(-rec_rate)
    ta = np.float32(wildcards.ta)
    nreps = np.int32(wildcards.nreps)
    na = np.int32(wildcards.na)
    n0 = np.int32(wildcards.n0)
    Ne = np.int32(wildcards.Ne)
    if wildcards.scenario == 'SerialConstant':
      cur_two_locus = TwoLocusSerialCoalescent(ta=ta, rec_rate=rec_rate, na=na, n0=n0, Ne=Ne, reps=nreps)
    elif wildcards.scenario == 'IBDNeUK10K':
      cur_two_locus = TwoLocusSerialIBDNeUK10K(ta=ta, rec_rate=rec_rate, na=na, n0=n0, reps=nreps, demo_file=ibd_ne_demo_file)
      cur_two_locus._set_demography()
    elif wildcards.scenario == 'Tennessen':
      cur_two_locus = TwoLocusSerialTennessen(ta=ta, n0=1,na=1, rec_rate=rec_rate, reps=nreps)
    elif wildcards.scenario == 'InstantGrowth7':
      cur_two_locus = TwoLocusSerialBottleneck(Ne=Ne, ta=ta, n0=1,na=1, Tstart=100, Tend=500000, Nbot=1e2, rec_rate=rec_rate, reps=nreps)
    elif wildcards.scenario == 'InstantGrowth8':
      cur_two_locus = TwoLocusSerialBottleneck(Ne=Ne, ta=ta, n0=1,na=1, Tstart=200, Tend=500000, Nbot=1e2, rec_rate=rec_rate, reps=nreps)
    elif wildcards.scenario == 'InstantGrowth9':
      cur_two_locus = TwoLocusSerialBottleneck(Ne=Ne, ta=ta, n0=1,na=1, Tstart=400, Tend=500000, Nbot=1e2, rec_rate=rec_rate, reps=nreps)
    else:
      raise ValueError('Improper value input for this simulation!')
    cur_two_locus._simulate()
    cur_two_locus._two_locus_branch_length()
    paired_branch_length = cur_two_locus.pair_branch_length / (2.*cur_two_locus.Ne)
    np.savez(str(output), paired_branch_length=paired_branch_length, Ne=cur_two_locus.Ne)


rule run_sims_all:
  input:
    expand(config['tmpdir'] + 'two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne10000.rep{nreps}.branch_length.npz', scenario=['Tennessen', 'IBDNeUK10K'], n0=1, na=1, ta=np.arange(0, 501, 50), rec_rate=4, nreps=50000),
    expand(config['tmpdir'] + 'two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne{Ne}.rep{nreps}.branch_length.npz', scenario=['SerialConstant'], n0=1, na=1, ta=np.arange(0, 501, 50), rec_rate=4, Ne=[5000, 10000, 20000], nreps=50000),
    expand(config['tmpdir'] + 'two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne{Ne}.rep{nreps}.branch_length.npz', scenario=['InstantGrowth7', 'InstantGrowth8', 'InstantGrowth9'], n0=1, na=1, ta=np.arange(0, 501, 50), rec_rate=4, Ne=[1000000], nreps=50000)
