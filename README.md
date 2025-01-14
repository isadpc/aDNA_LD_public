# aDNA LD Public

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aabiddanda/aDNA_LD_public/master)

The public facing repository for looking at two-locus and haplotype copying properties in models with temporal sampling.

## Figure Directory

If you are interested in a particular figure from the manuscript, the relevant iPython notebook is listed below:

**Main Figures**
* [Figure 1](notebooks/main_analyses/two_locus_adna.ipynb)
* [Figure 2](notebooks/main_analyses/two_locus_demographic.ipynb)
* [Figure 3](notebooks/main_analyses/two_locus_demographic.ipynb)
* [Figure 4](notebooks/main_analyses/corr_piA_piB_real_data.ipynb)
* [Figure 5](notebooks/main_analyses/joint_LDstats.ipynb)
* [Figure 6](notebooks/main_analyses/demography_copying_rate.ipynb)
* [Figure 7](notebooks/main_analyses/reich_1240k_analysis.ipynb)

**Supplementary Figures**

* [Figure S1](notebooks/supp_analyses/two_locus_adna_divergence.ipynb)
* [Figure S2](notebooks/main_analyses/two_locus_demographic.ipynb)
* [Figure S3](notebooks/main_analyses/two_locus_demographic.ipynb)
* [Figure S4](notebooks/supp_analyses/time_to_first_coal_anc_sample.ipynb)
* [Figure S5](notebooks/main_analyses/reich_1240k_analysis.ipynb)
* [Figure S6](notebooks/main_analyses/reich_1240k_analysis.ipynb)
* [Figure S7](notebooks/supp_analyses/check_first_coal.ipynb)

### Data

The data here represents intermediate data sources to generate the CSV files in `results`. These are typically in the form of tables that represent genetic map coordinates or sample names.

### Results

The `results` directory houses all of the files that are necessary to recreate the plots in both the main text and the supplementary materials. They represent the final output of `snakemake` rules that perform either simulations or estimate parameters from the data. If you are interested in the raw data used to generate the plots, this is where you want to take a look.

## Snakemake to recreate results

The files in the `snakefiles` directory are not directly used in this setting, but can be used in conjunction with `snakemake` to rerun the entire analysis and replicate our simulation results fully.

To re-run the full analysis (not using the pre-generated results):
you can run:

```
snakemake -s main.smk all_sim_results -j <number of cores>
```

Note that you will also want to change the `tmpdir` parameter in the `config.yml` file so that you have a place where you can write XXX Gb of data. Be warned that re-running all of the simulation analyses takes ~4-5 hours on a computing cluster with 200 parallel jobs (so is likely to take longer on a single laptop).

For our results on real ancient data, we have not chosen to store the data within this repository as it breaks some file-size limits on github, but have provided a fast `snakemake` rule to download the data from Dropbox and unpack it (~ 6 GB of data):

```
snakemake -s main.smk download_data -j <number of cores>
```

If you are interested in re-creating the results CSV files with the newly downloaded ancient male X-chromosome data:

```
snakemake -s main.smk infer_jump_rates_real_data_all -j <number of cores>
```

This recreation of the haplotype-copying inference data will also generally take quite some time (~10 hours on a computing cluster with 200 parallel jobs).

## Source Code

The `src` directory contains implementations of:

 1. Coalescent simulations with serial sampling using `msprime` (including two-locus simulations)
 2. A python-based implementation of the Li-Stephens model (using `numba`)
 3. Theoretical formulas for the correlation in tree-length and tree height across two loci in scenarios with serial sampling (`coal_cov.py`)


## Acknowledgements

* Matthias Steinrücken
* John Novembre
* Novembre, Steinrucken, Berg Labs @ UChicago
* NIH GRTG
