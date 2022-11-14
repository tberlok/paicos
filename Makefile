SHELL=bash

python:
	python setup.py build_ext --inplace

clean:
	python setup.py clean
	rm -rf paicos/cython/*.{c,so,html}
	rm -rf paicos/__pycache__ paicos/cython/__pycache__
