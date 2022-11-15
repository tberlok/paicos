from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import sys
import os
import numpy

if sys.platform == 'darwin':
    os.environ['CC'] = 'gcc-11'
    os.environ['CXX'] = 'g++-11'

Options.annotate = True
compiler_directives = {"boundscheck": False, "cdivision": True,
                       "wraparound": False, 'language_level': "3"}

# compiler_directives = {"boundscheck": True, "cdivision": True,
#                        "wraparound": True, 'language_level': "3"}
ext_modules = [
    Extension(
        "*",
        ["paicos/cython/*.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name='paicos',
    ext_modules=cythonize(ext_modules,
                          compiler_directives=compiler_directives))
