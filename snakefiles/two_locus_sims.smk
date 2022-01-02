"""Running two-locus simulations to check Monte-Carlo results."""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

# Import all of the functionality that we would need
sys.path.append("src/")
from coal_cov import *
from time_strat_ld import TimeStratifiedLDStats


# Import configurations to add some things
configfile: "config.yml"


# Data file for IBDNe UK10K demography
ibd_ne_demo_file = "data/demo_models/uk10k.IBDNe.txt"

## Approximating different sequence differences
# rho = 400 -> 1 Mb
# rho = 40 -> 100 kb
# rho = 4 -> 10 kb
rhos2 = [0.01, 0.1, 1, 10, 100, 500, 1000]

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
        config["tmpdir"]
        + "two_loci/serial/tmrca_{n0,\d+}_{na,\d+}/{rho}_{ta}_{nreps,\d+}_tmrca.npy",
    run:
        Ne = 1e4
        rho = np.float32(wildcards.rho)
        rec_rate = rho / (4.0 * Ne)
        ta = np.float32(wildcards.ta)
        nreps = np.int32(wildcards.nreps)
        na = np.int32(wildcards.na)
        n0 = np.int32(wildcards.n0)
        cur_two_locus = TwoLocusSerialCoalescent(
            ta=ta * 2.0 * Ne, Ne=Ne, rec_rate=rec_rate, na=na, n0=n0, reps=nreps
        )
        cur_two_locus._simulate()
        cur_two_locus._two_locus_tmrca()
        np.save(str(output), cur_two_locus.pair_tmrca)

        """
        Simulate two-locus branch lengths (useful for calculating empirical covariances!)
        """


rule sim_two_locus_branch_length:
    output:
        config["tmpdir"]
        + "two_loci/serial/branch_length_{n0, \d+}_{na,\d+}/{rho}_{ta}_{nreps,\d+}_branch_length.npy",
    run:
        Ne = 1e4
        rho = np.float32(wildcards.rho)
        rec_rate = rho / (4.0 * Ne)
        ta = np.float32(wildcards.ta)
        nreps = np.int32(wildcards.nreps)
        na = np.int32(wildcards.na)
        n0 = np.int32(wildcards.n0)
        cur_two_locus = TwoLocusSerialCoalescent(
            ta=ta * 2.0 * Ne, Ne=Ne, rec_rate=rec_rate, na=na, n0=n0, reps=nreps
        )
        cur_two_locus._simulate()
        cur_two_locus._two_locus_branch_length()
        np.save(str(output), cur_two_locus.pair_branch_length)


# ------- Finalized simulations under different demographic histories -------#
rule run_two_locus_sims_scenarios:
    output:
        config["tmpdir"]
        + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0,\d+}_na{na,\d+}.ta{ta,\d+}.r_{rec_rate}.Ne{Ne,\d+}.rep{nreps,\d+}.seed_{seed,\d+}.branch_length.npz",
    wildcard_constraints:
        scenario="SerialConstant|IBDNeUK10K|Tennessen|InstantGrowth[0-9]*|DivergenceMigration[0-9]*",
    run:
        rec_rate = np.float32(wildcards.rec_rate)
        rec_rate = 10 ** (-rec_rate)
        ta = np.float32(wildcards.ta)
        nreps = np.int32(wildcards.nreps)
        na = np.int32(wildcards.na)
        n0 = np.int32(wildcards.n0)
        Ne = np.int32(wildcards.Ne)
        if wildcards.scenario == "SerialConstant":
            cur_two_locus = TwoLocusSerialCoalescent(
                ta=ta, rec_rate=rec_rate, na=na, n0=n0, Ne=Ne, reps=nreps
            )
        elif wildcards.scenario == "IBDNeUK10K":
            cur_two_locus = TwoLocusSerialIBDNeUK10K(
                ta=ta,
                rec_rate=rec_rate,
                na=na,
                n0=n0,
                reps=nreps,
                demo_file=ibd_ne_demo_file,
            )
            cur_two_locus._set_demography()
        elif wildcards.scenario == "Tennessen":
            cur_two_locus = TwoLocusSerialTennessen(
                ta=ta, n0=1, na=1, rec_rate=rec_rate, reps=nreps
            )
        elif wildcards.scenario == "InstantGrowth7":
            cur_two_locus = TwoLocusSerialBottleneck(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                Tstart=100,
                Tend=500000,
                Nbot=1e2,
                rec_rate=rec_rate,
                reps=nreps,
            )
        elif wildcards.scenario == "InstantGrowth8":
            cur_two_locus = TwoLocusSerialBottleneck(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                Tstart=200,
                Tend=500000,
                Nbot=1e2,
                rec_rate=rec_rate,
                reps=nreps,
            )
        elif wildcards.scenario == "InstantGrowth9":
            cur_two_locus = TwoLocusSerialBottleneck(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                Tstart=400,
                Tend=500000,
                Nbot=1e2,
                rec_rate=rec_rate,
                reps=nreps,
            )
        elif wildcards.scenario == "DivergenceMigration1":
            cur_two_locus = TwoLocusSerialDivergence(
                Ne=Ne, ta=ta, n0=1, na=1, t_div=10, m=0.0, rec_rate=rec_rate, reps=nreps
            )
        elif wildcards.scenario == "DivergenceMigration2":
            cur_two_locus = TwoLocusSerialDivergence(
                Ne=Ne, ta=ta, n0=1, na=1, t_div=30, m=0.0, rec_rate=rec_rate, reps=nreps
            )
        elif wildcards.scenario == "DivergenceMigration3":
            cur_two_locus = TwoLocusSerialDivergence(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                t_div=30,
                m=1e-3,
                rec_rate=rec_rate,
                reps=nreps,
            )
        elif wildcards.scenario == "DivergenceMigration4":
            cur_two_locus = TwoLocusSerialDivergence(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                t_div=30,
                m=5e-3,
                rec_rate=rec_rate,
                reps=nreps,
            )
        elif wildcards.scenario == "DivergenceMigration5":
            cur_two_locus = TwoLocusSerialDivergence(
                Ne=Ne,
                ta=ta,
                n0=1,
                na=1,
                t_div=30,
                m=1e-2,
                rec_rate=rec_rate,
                reps=nreps,
            )
        else:
            raise ValueError("Improper value input for this simulation!")
        seed = np.int32(wildcards.seed)
        ts = cur_two_locus._simulate(random_seed=seed)
        cur_two_locus._two_locus_branch_length(ts)
        paired_branch_length = cur_two_locus.pair_branch_length
        # Saving the approach here ...
        np.savez(
            str(output),
            scenario=wildcards.scenario,
            seed=np.int32(wildcards.seed),
            rec_rate=np.float32(wildcards.rec_rate),
            ta=ta,
            paired_branch_length=paired_branch_length,
            Ne=cur_two_locus.Ne,
        )


rule run_sims_all:
    input:
        expand(
            config["tmpdir"]
            + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne{Ne}.rep{nreps}.seed_{seed}.branch_length.npz",
            scenario=["SerialConstant"],
            n0=1,
            na=1,
            ta=np.arange(0, 501, 20),
            seed=42,
            rec_rate=4,
            Ne=[5000, 10000, 20000],
            nreps=50000,
        ),
        expand(
            config["tmpdir"]
            + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne10000.rep{nreps}.seed_{seed}.branch_length.npz",
            scenario=["Tennessen", "IBDNeUK10K"],
            n0=1,
            na=1,
            ta=np.arange(0, 501, 20),
            seed=42,
            rec_rate=4,
            nreps=50000,
        ),
        expand(
            config["tmpdir"]
            + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne{Ne}.rep{nreps}.seed_{seed}.branch_length.npz",
            scenario=["InstantGrowth7", "InstantGrowth8", "InstantGrowth9"],
            n0=1,
            na=1,
            ta=np.arange(0, 501, 20),
            seed=42,
            rec_rate=4,
            Ne=[1000000],
            nreps=50000,
        ),


rule combine_branch_length_est:
    """Combine all of the branch length summary stats into a CSV that we can use later on."""
    input:
        files=rules.run_sims_all.input,
    output:
        "results/two_loci/multi_scenario_branch_len.csv",
    run:
        tot_df = []
        for x in tqdm(input.files):
            # Setting up a data frame entries
            df = np.load(x)
            Ne = df["Ne"]
            paired_bl = df["paired_branch_length"]
            ta = df["ta"]
            scenario = df["scenario"]
            seed = df["seed"]
            rec_rate = df["rec_rate"]
            # Compute statistics from the branch_lengths
            corr_bl = pearsonr(paired_bl[:, 0], paired_bl[:, 1])[0]
            cov_bl = np.cov(paired_bl[:, 0], paired_bl[:, 1])[0, 0]
            ebl = np.nanmean(paired_bl[:, 0])
            Ne_est = ebl / 2.0 / 2.0
            N = paired_bl.shape[0]
            cur_row = [
                scenario,
                ta,
                rec_rate,
                corr_bl,
                cov_bl,
                ebl,
                Ne,
                Ne_est,
                N,
                seed,
            ]
            tot_df.append(cur_row)
        # Creating the dataframe and outputting to a CSV
        df_final = pd.DataFrame(
            tot_df,
            columns=[
                "scenario",
                "ta",
                "rec_rate",
                "corr_bl",
                "cov_bl",
                "exp_bl",
                "Ne",
                "Ne_est",
                "Nreps",
                "seed",
            ],
        )
        df_final = df_final.dropna()
        df_final.to_csv(str(output), index=False, header=df_final.columns)


# ------- Two Locus Simulations for correlation in mutations ------- #
def pair_mut(ts_reps, nreps, **kwargs):
    """Computing the paired mutations in set of two-locus simulations"""
    pair_muts = np.zeros(shape=(nreps, 2))
    i = 0
    for ts in ts_reps:
        cur_muts = ts.segregating_sites(mode="site", windows=[0.0, 1.0, 2.0], **kwargs)
        pair_muts[i] = cur_muts[:2]
        i += 1
    return pair_muts


rule sim_two_locus_mutations:
    """Simulating the two locus mutations"""
    output:
        corr_est=config["tmpdir"]
        + "two_loci/serial/corrmut_theory/est_{ta,\d+}_theta{theta,\d+}_{nreps,\d+}_seed{seed,\d+}_corr_mut.npz",
    run:
        Ne = 1.0
        nreps = int(wildcards.nreps)
        ta = int(wildcards.ta) / 1000.0
        seed = int(wildcards.seed)
        rhos = np.logspace(-4, 2, 100)
        theta = int(wildcards.theta) / 1000.0
        corr_mut = np.zeros(rhos.size)
        for i, r in tqdm(enumerate(rhos)):
            cur_sim = TwoLocusSerialCoalescent(ta=ta, Ne=Ne, rec_rate=r, reps=nreps)
            ts_reps = cur_sim._simulate(mutation_rate=theta, random_seed=seed)
            m = pair_mut(ts_reps, nreps, span_normalise=False)
            corr_mut[i] = pearsonr(m[:, 0], m[:, 1])[0]
        se_r_mut = np.sqrt((1.0 - (corr_mut ** 2)) / (nreps - 2))
        # Saving the approach here ...
        np.savez(
            str(output),
            scenario="Two-Locus Theory",
            seed=np.int32(wildcards.seed),
            rec_rate=rhos,
            ta=ta,
            theta=theta,
            corr_piA_piB=corr_mut,
            nreps=nreps,
            se_corr_piApiB=se_r_mut,
            Ne=1.0,
        )


rule full_two_locus_sims:
    """Generating the full two-locus simulations for the correlation in number of mutations."""
    input:
        expand(
            config["tmpdir"]
            + "two_loci/serial/corrmut_theory/est_{ta}_theta{theta}_{nreps}_seed{seed}_corr_mut.npz",
            ta=[1000, 100, 10, 0],
            theta=400,
            nreps=[1000, 5000, nreps],
            seed=42,
        ),


rule collect_two_locus_sims:
    input:
        files=rules.full_two_locus_sims.input,
    output:
        "results/two_loci/theory_mut_corr.csv",
    run:
        tot_df = []
        for f in tqdm(input.files):
            # Load in the current dataset
            df = np.load(f)
            scenario = df["scenario"]
            rhos = df["rec_rate"]
            ta = df["ta"]
            theta = df["theta"]
            nreps = df["nreps"]
            corr_piA_piB = df["corr_piA_piB"]
            se_corr_piApiB = df["se_corr_piApiB"]
            seed = df["seed"]
            for (r, cr, se_r) in zip(rhos, corr_piA_piB, se_corr_piApiB):
                tot_df.append([scenario, ta, seed, r, cr, se_r, nreps])
        # Creating the dataframe and outputting to a CSV
        df_final = pd.DataFrame(
            tot_df,
            columns=[
                "scenario",
                "ta",
                "seed",
                "rec_rate",
                "corr_piApiB",
                "se_corr_piApiB",
                "nreps",
            ],
        )
        df_final = df_final.dropna()
        df_final.to_csv(str(output), index=False, header=df_final.columns)


# --------- Using two-locus simulations to simulate pairwise mutations --------- #
rule sim_two_locus_mut_haps:
    """Simulating the two locus haplotypes ... """
    output:
        corr_est=config["tmpdir"]
        + "two_loci/serial/ed0dt_norm/est_{ta,\d+}_theta{theta,\d+}_{nreps,\d+}_seed{seed,\d+}_ld_d0dtjt.npz",
    run:
        Ne = 1.0
        nreps = int(wildcards.nreps)
        ta = int(wildcards.ta) / 1000.0
        seed = int(wildcards.seed)
        rhos = np.logspace(-4, 2, 50)
        theta = int(wildcards.theta) / 1000.0
        ed0dt_norm_mean = np.zeros(rhos.size)
        ed0dt_norm_std = np.zeros(rhos.size)
        for i, r in tqdm(enumerate(rhos)):
            cur_ed0dt_norm = []
            cur_sim = TwoLocusSerialCoalescent(
                ta=ta, Ne=Ne, rec_rate=r, reps=nreps, na=500, n0=500
            )
            ts_reps = cur_sim._simulate(mutation_rate=theta, random_seed=seed + i)
            for ts in ts_reps:
                tree = ts.first()
                times = np.array([tree.time(i) for i in ts.samples()])
                pos = np.array([v.position for v in ts.variants()])
                gt = ts.genotype_matrix().T
                (
                    Dmod,
                    Danc,
                    norm_factor,
                    _,
                ) = TimeStratifiedLDStats.time_strat_two_locus_haps(
                    gt, pos, times, ta=ta, maf=0.05, polymorphic_total=False
                )
                ed0dt_norm = (Dmod * Danc) / norm_factor
                cur_ed0dt_norm.append(ed0dt_norm)
            cur_ed0dt_norm = np.hstack(cur_ed0dt_norm)
            ed0dt_norm_mean[i] = np.nanmean(cur_ed0dt_norm)
            ed0dt_norm_std[i] = np.nanstd(cur_ed0dt_norm)
        print(rhos)
        print(ed0dt_norm_mean)
        # Saving the approach here ...
        np.savez(
            str(output),
            scenario="Two-Locus Theory",
            seed=np.int32(wildcards.seed),
            rec_rate=rhos,
            ta=ta,
            theta=theta,
            ed0dtnorm=ed0dt_norm_mean,
            ed0dtnormstd=ed0dt_norm_std,
            nreps=nreps,
            Ne=1.0,
        )


rule sim_ld_time_strat_two_locus:
    input:
        expand(
            config["tmpdir"]
            + "two_loci/serial/ed0dt_norm/est_{ta}_theta{theta}_{nreps}_seed{seed}_ld_d0dtjt.npz",
            seed=42,
            nreps=[1],
            theta=1000,
            ta=[0],
        ),


# --------- Simulate two-locus mutational data under the more realistic model --------- #
rule sim_two_locus_mutations_european:
    """Simulating the two locus mutations"""
    output:
        corr_est=config["tmpdir"]
        + "two_loci/serial/corrmut_european/est_{ta,\d+}_theta{theta,\d+}_{nreps,\d+}_seed{seed,\d+}_corr_mut.npz",
    run:
        Ne = 1.0
        nreps = int(wildcards.nreps)
        ta = int(wildcards.ta)
        seed = int(wildcards.seed)
        rhos = np.logspace(-4, -2, 100)
        # NOTE: the theta variable here doesn't really matter ...
        theta = int(wildcards.theta) / 1000.0
        theta = 1e3 * 1.2e-8
        corr_mut = np.zeros(rhos.size)
        for i, r in tqdm(enumerate(rhos)):
            cur_sim = TwoLocusSerialTennessen(ta=ta, Ne=Ne, rec_rate=r, reps=nreps)
            ts_reps = cur_sim._simulate(mutation_rate=theta, random_seed=seed)
            m = pair_mut(ts_reps, nreps, span_normalise=False)
            corr_mut[i] = pearsonr(m[:, 0], m[:, 1])[0]
        se_r_mut = np.sqrt((1.0 - (corr_mut ** 2)) / (nreps - 2))
        # Saving the approach here ...
        np.savez(
            str(output),
            scenario="Two-Locus Tennessen",
            seed=np.int32(wildcards.seed),
            rec_rate=rhos,
            ta=ta,
            theta=theta,
            corr_piA_piB=corr_mut,
            nreps=nreps,
            se_corr_piApiB=se_r_mut,
            Ne=1.0,
        )


rule corr_mut_european:
    input:
        expand(
            config["tmpdir"]
            + "two_loci/serial/corrmut_european/est_{ta}_theta{theta}_{nreps}_seed{seed}_corr_mut.npz",
            ta=[0, 233, 1500],
            theta=400,
            nreps=10000,
            seed=range(1, 6),
        ),


rule collect_two_locus_sims_european:
    input:
        files=rules.corr_mut_european.input,
    output:
        "results/two_loci/european_real_data_mut_corr.csv",
    run:
        tot_df = []
        for f in tqdm(input.files):
            # Load in the current dataset
            df = np.load(f)
            print(df)
            scenario = df["scenario"]
            rhos = df["rec_rate"]
            ta = df["ta"]
            theta = df["theta"]
            nreps = df["nreps"]
            corr_piA_piB = df["corr_piA_piB"]
            se_corr_piApiB = df["se_corr_piApiB"]
            seed = df["seed"]
            for (r, cr, se_r) in zip(rhos, corr_piA_piB, se_corr_piApiB):
                tot_df.append([scenario, ta, seed, r, cr, se_r, nreps])
        # Creating the dataframe and outputting to a CSV
        df_final = pd.DataFrame(
            tot_df,
            columns=[
                "scenario",
                "ta",
                "seed",
                "rec_rate",
                "corr_piApiB",
                "se_corr_piApiB",
                "nreps",
            ],
        )
        df_final = df_final.dropna()
        df_final.to_csv(str(output), index=False, header=df_final.columns)


rule full_two_locus_bl_european:
    """Generating the full two-locus simulations for the correlation in number of mutations."""
    input:
        expand(
            config["tmpdir"]
            + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne10000.rep{nreps}.seed_{seed}.branch_length.npz",
            scenario=["Tennessen", "IBDNeUK10K"],
            n0=1,
            na=1,
            ta=[0, 233, 1500],
            seed=[42],
            rec_rate=np.around(np.linspace(3, 5, 30), 2),
            nreps=20000,
        ),


rule combine_branch_length_est_two_locus_european:
    """Combine all of the branch length summary stats into a CSV that we can use later on."""
    input:
        files=rules.full_two_locus_bl_european.input,
    output:
        "results/two_loci/tennessen_real_data_branch_len.csv",
    run:
        tot_df = []
        for x in tqdm(input.files):
            # Setting up a data frame entries
            df = np.load(x)
            Ne = df["Ne"]
            paired_bl = df["paired_branch_length"]
            ta = df["ta"]
            scenario = df["scenario"]
            seed = df["seed"]
            rec_rate = df["rec_rate"]
            # Compute statistics from the branch_lengths
            corr_bl = pearsonr(paired_bl[:, 0], paired_bl[:, 1])[0]
            cov_bl = np.cov(paired_bl[:, 0], paired_bl[:, 1])[0, 0]
            ebl = np.nanmean(paired_bl[:, 0])
            Ne_est = ebl / 2.0 / 2.0
            N = paired_bl.shape[0]
            cur_row = [
                scenario,
                ta,
                rec_rate,
                corr_bl,
                cov_bl,
                ebl,
                Ne,
                Ne_est,
                N,
                seed,
            ]
            tot_df.append(cur_row)
        # Creating the dataframe and outputting to a CSV
        df_final = pd.DataFrame(
            tot_df,
            columns=[
                "scenario",
                "ta",
                "rec_rate",
                "corr_bl",
                "cov_bl",
                "exp_bl",
                "Ne",
                "Ne_est",
                "Nreps",
                "seed",
            ],
        )
        df_final = df_final.dropna()
        df_final.to_csv(str(output), index=False, header=df_final.columns)


rule full_two_locus_bl_divergence_migration:
    """Generating the full two-locus simulations for the correlation in number of mutations."""
    input:
        expand(
            config["tmpdir"]
            + "two_loci/demographies/{scenario}/two_locus_sims_n0{n0}_na{na}.ta{ta}.r_{rec_rate}.Ne10000.rep{nreps}.seed_{seed}.branch_length.npz",
            scenario=[
                "DivergenceMigration1",
                "DivergenceMigration2",
                "DivergenceMigration3",
                "DivergenceMigration4",
                "DivergenceMigration5",
            ],
            n0=1,
            na=1,
            ta=[0, 233, 1500],
            seed=[42],
            rec_rate=np.around(np.linspace(3, 5, 30), 2),
            nreps=20000,
        ),


rule combine_branch_length_est_two_locus_migration_divergence:
    """Combine all of the branch length summary stats into a CSV that we can use later on."""
    input:
        files=rules.full_two_locus_bl_divergence_migration.input,
    output:
        "results/two_loci/divergence_migration_branch_len.csv",
    run:
        tot_df = []
        for x in tqdm(input.files):
            # Setting up a data frame entries
            df = np.load(x)
            Ne = df["Ne"]
            paired_bl = df["paired_branch_length"]
            ta = df["ta"]
            scenario = df["scenario"]
            seed = df["seed"]
            rec_rate = df["rec_rate"]
            # Compute statistics from the branch_lengths
            corr_bl = pearsonr(paired_bl[:, 0], paired_bl[:, 1])[0]
            cov_bl = np.cov(paired_bl[:, 0], paired_bl[:, 1])[0, 0]
            ebl = np.nanmean(paired_bl[:, 0])
            Ne_est = ebl / 2.0 / 2.0
            N = paired_bl.shape[0]
            cur_row = [
                scenario,
                ta,
                rec_rate,
                corr_bl,
                cov_bl,
                ebl,
                Ne,
                Ne_est,
                N,
                seed,
            ]
            tot_df.append(cur_row)
        # Creating the dataframe and outputting to a CSV
        df_final = pd.DataFrame(
            tot_df,
            columns=[
                "scenario",
                "ta",
                "rec_rate",
                "corr_bl",
                "cov_bl",
                "exp_bl",
                "Ne",
                "Ne_est",
                "Nreps",
                "seed",
            ],
        )
        df_final = df_final.dropna()
        df_final.to_csv(str(output), index=False, header=df_final.columns)
