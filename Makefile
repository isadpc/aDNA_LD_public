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

# TODO : I need to have a targeted rule to run the notebook from top to bottom before generating a webpage...
