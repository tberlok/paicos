SHELL=bash

python:
	python3 setup.py build_ext --inplace

clean:
	python3 setup.py clean
	rm -rf paicos/cython/*.{c,so,html}
	rm -rf paicos/__pycache__ paicos/cython/__pycache__

cleanup:
	rm -r test_data
	rm data/very_small_snap_247.hdf5
	rm data/reduced_snap2_247.hdf5
