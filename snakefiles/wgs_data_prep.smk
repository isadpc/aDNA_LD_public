#!python3

"""
    Processing of ancient WGS data for modern comparisons
"""
import gzip as gz
import os
import sys
import numpy as np
import allel
import pandas as pd

# Individuals who we want for our ancient samples
anc_indivs = ['Loschbour', 'LBK', 'UstIshim']

# Choosing modern french individuals
mod_indivs_damgaard = ['SS6004468', 'SS6004474']
mod_indivs = ['LP6005441-DNA_A05', 'LP6005441-DNA_B05', 'SS6004468']
mod_east_asian = ['SS6004469', 'LP6005441-DNA_D05', 'LP6005441-DNA_C05']
mod_africa = ['SS6004475', 'LP6005442-DNA_B02', 'LP6005442-DNA_A02']

#Directory of current data currently 
data_dir = '/home/abiddanda/novembre_lab2/old_project/share/botai/data/vcfs/'
sgdp_dir = '/home/abiddanda/novembre_lab2/data/external_public/sgdp/merged/'
thousand_genomes_dir = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/haps/'
#1000 BIALLELIC LOCATIONS 
KGP_BIALLELIC = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.biallelic_snps.vcf.gz'

#recombination rate files that we want to use 
recomb_rate_dir = 'data/maps_b37/'
rec_bins = np.arange(0.01, 1.01, 0.01)
rec_map_types = {"Physical_Pos" : np.uint64, 
                 "deCODE" : np.float32, 
                 "COMBINED_LD" : np.float32, 
                 "YRI_LD": np.float32, 
                 "CEU_LD" : np.float32, 
                 "AA_Map": np.float32, 
                 "African_Enriched" : np.float32,
                 "Shared_Map" : np.float32}


# Definition for the autosomes...
AUTOSOMES = np.arange(1,23)

### ----- Rules to setup the ancient datasets   ----- ###
rule extract_autosomes:
  '''
    Extract autosomal samples only for the ancient samples we designate
  '''
  input:
    data_dir + 'samtools.combined.chr{CHROM}.release1.rn.vcf.gz'
  output:
    vcf='data/raw_data/tmp_staging/ancient_data.chr{CHROM,\d+}.damgaard.vcf.gz',
    idx='data/raw_data/tmp_staging/ancient_data.chr{CHROM,\d+}.damgaard.vcf.gz.tbi'
  threads: 4
  run:
    indiv_str = ','.join(anc_indivs) + ',' +  ','.join(mod_indivs_damgaard)
    shell('bcftools view --threads 4 -v snps -m2 -M2 -s {indiv_str} {input} | bcftools annotate -x INFO,^FORMAT/GT |  bgzip -@4 > {output.vcf}')
    shell('tabix {output.vcf}')

rule sgdp_ancient_merge:
  input:
    anc_vcf = rules.extract_autosomes.output.vcf,
    sgdp_vcf = sgdp_dir + 'sgdp_total_merged_chr{CHROM}.vcf.gz'
  output:
    merged_vcf = 'data/real_data_autosomes/merged_sgdp/ancient_sgdp_total_merged.chr{CHROM}.vcf.gz',
    merged_vcf_idx = 'data/real_data_autosomes/merged_sgdp/ancient_sgdp_total_merged.chr{CHROM}.vcf.gz.tbi'
  threads: 4
  run:
    """
    bcftools merge {input.anc_vcf} {input.sgdp_vcf} --force-samples | bcftools
    view -v snps -m2 -M2 | bcftools annotate -x INFO,^FORMAT/GT | bgzip -@4 > {output.merged_vcf}
    tabix -f {output.merged_vcf}
    """

rule kg_phase3_ancient_merge:
  """
    Merge the ancient data with data from the 1000 Genomes Phase 3 project (phased data!)
  """
  input:
    anc_vcf = rules.extract_autosomes.output.vcf,
    kg_phase3_vcf = thousand_genomes_dir + 'ALL.chr{CHROM}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
  output:
    merged_vcf = 'data/real_data_autosomes/merged_kgp/ancient_kgp_total_merged.chr{CHROM,\d+}.vcf.gz',
    merged_vcf_idx = 'data/real_data_autosomes/merged_kgp/ancient_kgp_total_merged.chr{CHROM,\d+}.vcf.gz.tbi'
  threads: 4
  shell:
    """
    bcftools merge -0 {input.anc_vcf} {input.kg_phase3_vcf} -Ou | bcftools view -c 1:minor -m2 -M2 -v snps | bcftools annotate -x INFO,^FORMAT/GT | bgzip -@4 > {output.merged_vcf}
    tabix -f {output.merged_vcf}
    """

rule merge_ancient_modern_all:
  """
    Final rule to generate full data 
  """
  input:
    expand('data/real_data_autosomes/merged_{x}/ancient_{x}_total_merged.chr{CHROM}.vcf.gz', CHROM=np.arange(1,23), x=['kgp'])


