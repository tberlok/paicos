SHELL=bash

python:
	python3 setup.py build_ext --inplace

clean:
	python3 setup.py clean
	rm -rf paicos/cython/*.{c,so,html}
	rm -rf paicos/__pycache__ paicos/cython/__pycache__
