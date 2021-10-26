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
anc_indivs = ['Loschbour', 'LBK', 'UstIshim','8.4_BGI_WG', 'Yamnaya']

# Choosing modern french individuals
mod_indivs_damgaard = ['HGDP00456', 'HGDP00521', 'HGDP00542', 'HGDP00665',
'HGDP00778', 'HGDP00877', 'HGDP00927', 'HGDP00998', 'HGDP01029',
'SS6004467', 'SS6004468', 'SS6004469', 'SS6004470', 'SS6004471', 
'SS6004472', 'SS6004473', 'SS6004474', 'SS6004475', 'SS6004476',
'SS6004477', 'SS6004478', 'SS6004479', 'SS6004480']

#Directory of current data currently 
data_dir = '/home/abiddanda/novembre_lab2/old_project/share/botai/data/vcfs/'
thousand_genomes_dir = '/home/abiddanda/novembre_lab2/data/external_public/1kg_phase3/haps/'

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
    stats = 'data/raw_data/tmp_staging/ancient_data.chr{CHROM,\d+}.damgaard.stats.gz',
    vcf = 'data/raw_data/tmp_staging/ancient_data.chr{CHROM,\d+}.damgaard.vcf.gz',
    idx = 'data/raw_data/tmp_staging/ancient_data.chr{CHROM,\d+}.damgaard.vcf.gz.tbi'
  threads: 4
  run:
    indiv_str = ','.join(anc_indivs) + ',' +  ','.join(mod_indivs_damgaard)
    shell('bcftools stats -s - {input} | gzip > {output.stats}')
    shell('bcftools view --threads 4 -s {indiv_str} {input} | bcftools annotate -x INFO,^FORMAT/GT |  bgzip -@4 > {output.vcf}')
    shell('tabix {output.vcf}')

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
    bcftools merge -0 {input.anc_vcf} {input.kg_phase3_vcf} -Ou | bcftools view -m2 -M2 -v snps | bcftools annotate -x INFO,^FORMAT/GT | bgzip -@4 > {output.merged_vcf}
    tabix -f {output.merged_vcf}
    """

rule merge_ancient_modern_all:
  """
    Final rule to generate full data from the merge 
  """
  input:
    expand('data/real_data_autosomes/merged_{x}/ancient_{x}_total_merged.chr{CHROM}.vcf.gz', CHROM=range(1,23), x=['kgp'])

