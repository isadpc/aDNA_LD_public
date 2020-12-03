# Import configurations
configfile: "config.yml"


include: "snakefiles/hap_copying.smk"
include: "snakefiles/two_locus_sims.smk"
include: "snakefiles/corr_seg_sites.smk"



rule get_raw_data:
  input:
    'xxx'



rule all_results:
  input:
    rules.concatenate_hap_copying_results.output,
    rules.concatenate_hap_copying_results_chrX_sim.output,
    rules.monte_carlo_sA_sB_results.output,
    rules.collapse_est_ta_Ne.output,
    rules.combine_branch_length_est.output,
    rules.concatenate_tot_corr_piA_piB.output,
    
