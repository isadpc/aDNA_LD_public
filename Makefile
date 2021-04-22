notebooks := $(wildcard notebooks/*.ipynb) $(wildcard notebooks/**/*.ipynb)
html_pages := $(patsubst notebooks/%.ipynb,docs/%.html,$(notebooks))

build.site: $(html_pages)
clean.site: ; rm $(html_pages)

print-%: ; @echo $* is $($*):

docs/%.html: notebooks/%.ipynb
	jupyter nbconvert\
		--to html $<\
		--output-dir $(dir $@)\
		--template classic\

#0. Re-run ALL analysis notebooks
all_analyses: main_analyses supp_analyses

#1. Main analysis notebooks
main_analyses: corr_piA_piB_real_data demography_copying_rate gen_corrbl_tables joint_ldstats two_locus_adna two_locus_demographic

corr_piA_piB_real_data:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/corr_piA_piB_real_data.ipynb

demography_copying_rate:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/demography_copying_rate.ipynb

gen_corrbl_tables:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/gen_corrbl_tables.ipynb

joint_ldstats:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/joint_LDstats.ipynb

reich1240k_analysis:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/reich_1240k_analysis.ipynb

two_locus_adna:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/two_locus_adna.ipynb

two_locus_demographic:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/main_analyses/two_locus_demographic.ipynb

#2. Supplementary Analysis notebooks
supp_analyses: check_first_coal est_ta_Ne time_to_first_coal two_locus_divergence validate_equations ls_verify

check_first_coal:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/check_first_coal.ipynb

est_ta_Ne:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/est_ta_Ne.ipynb

time_to_first_coal:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/time_to_first_coal_anc_sample.ipynb

two_locus_divergence:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/two_locus_adna_divergence.ipynb

validate_equations:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/validate_equations.ipynb

ls_verify:
	jupyter nbconvert --to notebook\
	 --execute --inplace notebooks/supp_analyses/ls_verify.ipynb
