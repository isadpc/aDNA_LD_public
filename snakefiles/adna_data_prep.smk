#!python3

"""
    Extracting Male X-Chromosome data from Reich et al. 
"""

import os
import sys
import numpy as np
import allel
import pandas as pd
from tqdm import tqdm

# Defining relevant directories for data analysis
REICH_LAB_1240K_DATA_DIR = 'data/raw_data/reich_lab_1240k_chrX_males/'
RECOMB_RATE_DIR= 'data/recmaps/'
DATA_DIR_1KG = 'data/raw_data/1kg_chrX_males/'

rec_map_types = {"Physical_Pos" : np.uint64,
                 "deCODE" : np.float32,
                 "COMBINED_LD" : np.float32,
                 "YRI_LD": np.float32,
                 "CEU_LD" : np.float32,
                 "AA_Map": np.float32,
                 "African_Enriched" : np.float32,
                 "Shared_Map" : np.float32}

rule filt_male_1240k_x_chrom:
  """
    Filter to only males on the x-chromosomes for Reich lab VCF
  """
  input:
    vcf_file = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.vcf.gz',
    ind_file = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.ind',
    chrom_rename = REICH_LAB_1240K_DATA_DIR + 'rename_x_chrom.txt'
  output:
    male_inds = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.males_only.ind',
    vcf_male_only = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.male_only.vcf.gz',
    vcf_male_only_idx = REICH_LAB_1240K_DATA_DIR + 'v42.4.1240K.chrX.male_only.vcf.gz.tbi',
  shell:
    """
    awk \'$2 == \"M\" {{print $1}}\' {input.ind_file} > {output.male_inds}
    bcftools view -S {output.male_inds} --force-samples {input.vcf_file} | bcftools annotate --rename-chrs {input.chrom_rename} | bgzip -@4 > {output.vcf_male_only}
    tabix -f {output.vcf_male_only}
    """

# --------- Using the OG 1000 Genomes Phase 3 as a panel ----------- #
rule generate_chrX_1kg_panel_vcf:
  input:
    chrX_vcf_1kg = DATA_DIR_1KG + 'ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz',
    panel_indivs = 'data/hap_copying/chrX_male_analysis/panel_files/{panel}.real_1kg.male_only.indivs.txt',
    reich_lab_chrX_1240K_male_only_vcf = rules.filt_male_1240k_x_chrom.output.vcf_male_only,
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)',
    panel = '(ceu|eur|fullkg)'
  output:
    tmp_chrX_pos = temp('data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.pos'),
    vcf = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.vcf.gz',
    vcf_idx = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/chrX.{panel}.real_1kg.male_only.vcf.gz.tbi'
  shell:
    """
      bcftools query -f \'%CHROM\t%POS\n\' {input.reich_lab_chrX_1240K_male_only_vcf} > {output.tmp_chrX_pos}
      bcftools view -v snps -m2 -M2 -S {input.panel_indivs} -R {output.tmp_chrX_pos} {input.chrX_vcf_1kg} | bcftools sort | bgzip -@5 > {output.vcf}
      tabix -f {output.vcf}
    """

rule merge_real_1kg_vcf_chrX_1240K_panel:
  input:
    vcf_1240k_only = rules.filt_male_1240k_x_chrom.output.vcf_male_only,
    vcf_real_1kg =  rules.generate_chrX_1kg_panel_vcf.output.vcf,
  output:
    tmp_chrX_pos = temp('data/hap_copying/chrX_male_analysis/vcf_real1kg/merged/chrX.{panel}.real_1kg.v42_merged.1240K_only.male_only.sites.pos'),
    vcf = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/merged/chrX.{panel}.real_1kg.v42_merged.1240K_only.male_only.vcf.gz',
    vcf_idx = 'data/hap_copying/chrX_male_analysis/vcf_real1kg/merged/chrX.{panel}.real_1kg.v42_merged.1240K_only.male_only.vcf.gz.tbi'
  shell:
    """
      bcftools query -f \"%CHROM\t%POS\n\" {input.vcf_real_1kg} > {output.tmp_chrX_pos}
      bcftools merge {input.vcf_1240k_only} {input.vcf_real_1kg} -R {output.tmp_chrX_pos} | bgzip -@4 > {output.vcf}
      tabix -f {output.vcf}
    """

rule interp_gen_hap_panel_chrX_real1kg:
  """
     Linearly interpolate recombination position into the full X-chromosome panel
     from the 1000 Genomes Project Phase 3 data
  """
  input:
    vcf = rules.merge_real_1kg_vcf_chrX_1240K_panel.output.vcf,
    vcf_idx = rules.merge_real_1kg_vcf_chrX_1240K_panel.output.vcf_idx,
    genmap = RECOMB_RATE_DIR + 'maps_chr.X'
  output:
    panel_file='data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.{panel}.real_1kg.chrX.male_only.recmap_{rec}.total.npz'
  wildcard_constraints:
    rec = '(CEU_LD|deCODE)',
    panel = '(ceu|eur|fullkg)'
  run:
    # Reading in VCF
    vcf_data = allel.read_vcf(input.vcf)
    geno = vcf_data['calldata/GT']
    # Taking the first haplotype here since it should be equivalent ...
    geno_summed = geno[:,:,0]
    geno_summed = geno_summed.astype(np.int8)
    geno_summed[geno_summed == 2] = 1
    # Generating the list of positions
    pos = vcf_data['variants/POS']
    # Reference Alleles
    ref_alleles = vcf_data['variants/REF']
    # Alternative Alleles
    alt_alleles = vcf_data['variants/ALT']
    sample_IDs = vcf_data['samples']
    # Reading in the recombination map
    rec_df = pd.read_csv(input.genmap, sep='\s+', low_memory=True, dtype=rec_map_types)
    rec_pos = rec_df['Physical_Pos'].values
    rec_dist = rec_df[str(wildcards.rec)].values
    subsequent_distance = rec_dist[:-1] - rec_dist[1:]
    idx = np.where(subsequent_distance != 0)[0]
    min_pos, max_pos = rec_pos[np.min(idx)], rec_pos[np.max(idx)]
    # Filter to the minimul
    filt_idx = np.where((pos <= max_pos) & (pos >= min_pos))[0]
    real_pos_filt = pos[filt_idx]
    gt_filt = geno_summed[filt_idx,:]
    ref_alleles = ref_alleles[filt_idx]
    alt_alleles = alt_alleles[filt_idx]
    interp_rec_pos = np.interp(real_pos_filt, rec_pos[idx], rec_dist[idx])
    # Saving the file
    np.savez_compressed(output.panel_file, gt=gt_filt, samples=sample_IDs, ref=ref_alleles, alt=alt_alleles, bp_pos = real_pos_filt, cm_pos = interp_rec_pos)

# Final rule to generate all of the datasets that we will end up using ...
rule gen_all_aDNA_datasets:
  input:
    expand('data/hap_copying/chrX_male_analysis/tot_chrX_panel/tot_chrX.{panel}.real_1kg.chrX.male_only.recmap_{rec}.total.npz', panel=['ceu','eur','fullkg'], rec='deCODE')
