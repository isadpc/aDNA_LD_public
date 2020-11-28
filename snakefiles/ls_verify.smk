"""
  Snakemake file to verify estimation of the jump-rate from the LS-model in simulations.
"""

import os
import sys
import numpy as np
import msprime as msp
import tszip
import allel
import pandas as pd
from tqdm import tqdm

# Import all of the functionality that we would need
sys.path.append('src/')
from li_stephens import *
from aDNA_coal_sim import *

# Import configurations to add some things
configfile: "config.yml"

def ascertain_variants(hap_panel, pos, maf=0.05):
    """Ascertaining variants based on frequency."""
    assert((maf < 0.5) & (maf > 0))
    mean_daf = np.mean(hap_panel, axis=0)
    idx = np.where((mean_daf > maf) | (mean_daf < (1. - maf)))[0]
    asc_panel = hap_panel[:,idx]
    asc_pos = pos[idx]
    return(asc_panel, asc_pos, idx)


# ------- 1. Generate a modern haplotype panel from ChrX using the deCODE map ------- #
# - TODO : should we implement ascertainment as well.
rule gen_hap_panel_real_map:
  input:
    recmap = config['recmaps']['deCODE']['chrX']
  output:
    hap_panel = config['tmpdir'] + 'ls_verify/data/chrX_{n, \d+}_{rep,\d+}.panel.npz'
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
    cm_pos_filt = np.interp(pos_filt, df_recmap['Physical_Pos'].values, df_recmap['deCODE'].values)
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



# ------- 2. Simulating & Inferring from the LS model ------ #
rule infer_scale_real_map:
  input:
    hap_panel = rules.gen_hap_panel_real_map.output.hap_panel
  output:
    scale_inf = config['tmpdir'] + 'ls_verify/results/samp_{n,\d+}.scale_{scale_min,\d+}_{scale_max,\d+}.seed{seed,\d+}.rep{rep,\d+}.npz'
  run:
    scale_min, scale_max = int(wildcards.scale_min), int(wildcards.scale_max)
    assert(scale_min < scale_max)
    assert(scale_min % 100 == 0)
    assert(scale_max % 100 == 0)
    scales_true = np.arange(scale_min, scale_max, 100)
    df = np.load(input.hap_panel)
    ls_model = LiStephensHMM(df['haps'], df['rec_pos'])
    # Setting up the test haplotypes
    test_haps = [ls_model.sim_haplotype(scale=s, eps=1e-2, seed=int(wildcards.seed))[0] for s in scales_true]
    # Setting result directories ...
    scales_marg_hat = np.zeros(scales_true.size)
    scales_jt_hat = np.zeros(scales_true.size)
    eps_jt_hat = np.zeros(scales_true.size)
    se_scales_jt_hat = np.zeros(scales_true.size)
    se_eps_jt_hat = np.zeros(scales_true.size)
    # Iterate through all of these sequentially
    for i in tqdm(range(scales_true.size)):
      # Inferring the marginal
      res = ls_model.infer_scale(test_haps[i], eps=1e-2, bounds=(1.,1e6), method='bounded', tol=1e-7, options={'disp':3})
      scales_marg_hat[i] = res.x
      # Inferring the joint values
      res_jt = ls_model.infer_params(test_haps[i], x0=[1e2,1e-3], bounds=[(1.,1e7), (1e-6,0.2)], tol=1e-7)
      scales_jt_hat[i] = res_jt.x[0]
      eps_jt_hat[i] = res_jt.x[1]
      se_scales_jt_hat[i] = np.sqrt(res_jt.hess_inv.todense()[0,0])
      se_eps_jt_hat[i] = np.sqrt(res_jt.hess_inv.todense()[1,1])
    # Saving the output file
    np.savez(output.scale_inf,
             scales_true=scales_true,
             scales_marg_hat=scales_marg_hat,
             scales_jt_hat=scales_jt_hat,
             eps_jt_hat=eps_jt_hat,
             se_scales_jt_hat=se_scales_jt_hat,
             se_eps_jt_hat=se_eps_jt_hat)


# ------- 3. Combine all of the estimates into a spreadsheet -------- #
rule combine_ls_verify:
  input:
    data = lambda wildcards: expand(config['tmpdir'] + 'ls_verify/results/samp_{n}.scale_{scale_min}_{scale_max}.seed{seed}.rep{rep}.npz', n=wildcards.n, rep=np.arange(20), scale_min=100, scale_max=1000, seed=42)
  output:
    csv = 'results/ls_verify/ls_simulations_{n,\d+}.csv'
  run:
    # Read through all of the sim results and concatenate
    scales_true = []
    scales_marg_hat = []
    scales_jt_hat = []
    eps_jt_hat = []
    se_scales_jt_hat = []
    se_eps_jt_hat = []
    for f in input.data:
      df = np.load(f)
      scales_true.append(df['scales_true'])
      scales_marg_hat.append(df['scales_marg_hat'])
      scales_jt_hat.append(df['scales_jt_hat'])
      eps_jt_hat.append(df['eps_jt_hat'])
      se_scales_jt_hat.append(df['se_scales_jt_hat'])
      se_eps_jt_hat.append(df['se_eps_jt_hat'])
    # concatenate all of the numpy arrays
    scales_true = np.hstack(scales_true)
    scales_marg_hat = np.hstack(scales_marg_hat)
    scales_jt_hat = np.hstack(scales_jt_hat)
    eps_jt_hat = np.hstack(eps_jt_hat)
    se_scales_jt_hat = np.hstack(se_scales_jt_hat)
    se_eps_jt_hat = np.hstack(se_eps_jt_hat)
    # Make it into a dataframe
    d = {'scales_true': scales_true,
         'scales_marg_hat':scales_marg_hat,
         'scales_jt_hat':scales_jt_hat,
         'eps_jt_hat':eps_jt_hat,
         'se_scales_jt_hat':se_scales_jt_hat,
         'se_eps_jt_hat': se_eps_jt_hat}
    df = pd.DataFrame(d)
    df.to_csv(output.csv, index=False)



rule full_verify:
  input:
    expand('results/ls_verify/ls_simulations_{n}.csv', n=[100])
