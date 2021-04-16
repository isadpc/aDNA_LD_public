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

# TODO : get the automatic running of notebooks to work...
#notebooks/*.ipynb:
#    jupyter nbconvert --execute --to notebook --inplace $@