PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CYTHONSRC=$(wildcard pyearth/*.pyx)
CSRC=$(CYTHONSRC:.pyx=.c)

inplace: cython
	$(PYTHON) setup.py build_ext -i

all: inplace

cython: $(CSRC)

clean:
	rm -f pyearth/*.c pyearth/*.so pyearth/*.pyc pyearth/test/*.pyc pyearth/test/basis/*.pyc pyearth/test/record/*.pyc

%.c: %.pyx
	$(CYTHON) $<

test: inplace
	$(NOSETESTS) -s pyearth

test-coverage: inplace
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage --cover-package=pyearth pyearth

verbose-test: inplace
	$(NOSETESTS) -sv pyearth

conda:
	conda-build conda-recipe
