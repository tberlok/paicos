import setuptools
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import Cython
import sys
import os
import numpy
import glob


# On MacOs, you will need to explicitly set the compiler to one that features openmp.
# Either by uncommenting below or by compiling with e.g. 'CC=gcc-14 make'
# if sys.platform == 'darwin':
#     os.environ['CC'] = 'gcc-13'
#     os.environ['CXX'] = 'g++-13'

Options.annotate = True
compiler_directives = {"boundscheck": False, "cdivision": True, "nonecheck": False,
                       "wraparound": False, 'language_level': "3"}
# compiler_directives = {"boundscheck": True, "cdivision": True,
#                        "wraparound": False, 'language_level': "3"}


include_dirs = ['paicos/cython/', numpy.get_include()]
extra_compile_args = ['-fopenmp', "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                      "-Wno-unused-function", '-O3', '-ffast-math', '-march=native']
extra_link_args = ['-fopenmp']
ext_modules = [
    Extension(
        name='paicos.cython.get_index_of_region',
        sources=['paicos/cython/get_index_of_region.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.cython.histogram',
        sources=['paicos/cython/histogram.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.cython.sph_projectors',
        sources=['paicos/cython/sph_projectors.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.cython.openmp_info',
        sources=['paicos/cython/openmp_info.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.cython.get_derived_variables',
        sources=['paicos/cython/get_derived_variables.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.trees.bvh_tree',
        sources=['paicos/trees/bvh_tree.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
    Extension(
        name='paicos.cython.ray_tracer',
        sources=['paicos/cython/ray_tracer.pyx'],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

install_requires = ['scipy',
                    'numpy',
                    'h5py',
                    'astropy',
                    'numba',
                    'matplotlib']


cur = os.path.abspath(__file__).replace('setup.py', '')
with open(cur + 'README.md') as f:
    long_description = f.read()

with open(cur + 'dev_requirements.txt') as f:
    dev_requirements = f.read().strip().split('\n')

setup(
    name='paicos',
    version='0.1.16',
    description=('An object-oriented Python package for analysis of '
                 + '(cosmological) simulations performed with Arepo.'),
    url='https://github.com/tberlok/paicos',
    author='Thomas Berlok',
    author_email='tberlok@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD 3-clause',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={'dev': dev_requirements},
    package_data={'paicos/cython': ['*.c', '*.so']},
    classifiers=['Programming Language :: Python :: 3'],
    ext_modules=cythonize(ext_modules,
                          compiler_directives=compiler_directives),
    cmdclass={'build_ext': Cython.Build.build_ext})
