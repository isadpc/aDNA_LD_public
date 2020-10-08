# aDNA LD Public

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aabiddanda/aDNA_LD_public/master)

The public facing repository for looking at two-locus and haplotype copying properties in models with temporal sampling

## Notebooks

For each section of the paper, there is a notebook that generates all relevant figures for that section of the manuscript. Typically these figures will be deposited in the `plots` directory under either `main_figs` or `supp_figs`.

## Data

The data here represents intermediate data sources to generate the plots in the `notebooks`. These are typically in the form of CSV files that are used similarly as supplementary tables as well.

## Snakemake

The files in the `snakefiles` directory are not directly used in this setting, but can be used in conjunction with `snakemake` to rerun the entire raw analysis and replicate our simulation results entirely.

## Source Code

The `src` directory contains implementations of:

 1. Coalescent simulations with serial sampling
 2. A python-based implementation of the Li-Stephens Model (using `numba`)
 3. Theoretical formulas for the correlation in tree-length and tree height across two loci in scenarios with serial sampling.
