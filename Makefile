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
	rm -rf test_data
	rm -f data/very_small_snap_247.hdf5
	rm -f data/reduced_snap2_247.hdf5

checks:
	make cleanup
	pytest tests/comoving
	pytest tests/non-comoving
	pytest tests/general

linting:
	flake8 ./
	pylint --errors-only --ignored-modules=astropy.units,astropy.constants  --disable=E0611,E0401,E1101,E1120,E1121 paicos

make dev_checks:
	make checks
	make linting

# Tests that can only run on some systems
make gpu_checks:
	python tests/cuda-gpu/test_gpu_binary_tree.py
	python tests/cuda-gpu/test_gpu_ray_projector.py
	python tests/cuda-gpu/test_gpu_sph_projector.py
