ROOTDIR = $(CURDIR)

.PHONY: Rpack Rbuild Rinstall Rcheck

# Script to make a clean installable R package.
Rpack:
	rm -rf StableTrees StableTrees*.tar.gz
	cp -r R-package StableTrees
	cp -r include StableTrees/inst
	cp ./LICENSE StableTrees

R ?= R

Rbuild: Rpack
	$(R) CMD build StableTrees
	rm -rf StableTrees
	
Rinstall: Rpack
	$(R) CMD install StableTrees*.tar.gz

Rcheck: Rbuild
	$(R) CMD check --as-cran StableTrees*.tar.gz