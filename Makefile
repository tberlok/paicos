SHELL=bash

python:
	python3 setup.py build_ext --inplace

clean:
	python3 setup.py clean
	rm -rf paicos/cython/*.{c,so,html}
	rm -rf __pycache__ paicos/__pycache__ paicos/cython/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf paicos.egg-info

cleanup:
	rm -rf data/test_data
	rm -f data/very_small_snap_247.hdf5
	rm -f data/reduced_snap2_247.hdf5

checks:
	make cleanup
	cd tests; pytest comoving
	cd tests; pytest non-comoving
	cd tests; pytest general
	cd tests; pytest cuda-gpu

linting:
	flake8 ./
	pylint --errors-only --ignored-modules=astropy.units,astropy.constants  --disable=E0606,0611,E0401,E1101,E1111,E1120,E1121,E1123 paicos

dev_checks:
	make checks
	make linting

docs:
	sphinx-apidoc -f -o docs/source .; cd docs; make html
