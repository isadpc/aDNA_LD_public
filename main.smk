# Import configurations
configfile: "config.yml"


include: "snakefiles/hap_copying.smk"
include: "snakefiles/hap_copying_real_data.smk"
include: "snakefiles/two_locus_sims.smk"
include: "snakefiles/corr_seg_sites.smk"
include: "snakefiles/ls_verify.smk"


rule get_raw_data:
  """Get the raw data from Dryad."""
  input:
    'xxx'


rule ls_verify_all:
  """Rule to generate results verifying."""
  input:
    rules.full_verify.input


rule data_sim_results:
  input:
    rules.concatenate_hap_copying_results.output,
    rules.concatenate_hap_copying_results_chrX_sim.output,
    rules.monte_carlo_sA_sB_results.output,
    rules.collapse_est_ta_Ne.output,
    rules.combine_branch_length_est.output,
    rules.concatenate_tot_corr_piA_piB.output,
    
