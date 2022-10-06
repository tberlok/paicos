from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import sys
import os

if sys.platform == 'darwin':
    os.environ['CC'] = 'gcc-11'
    os.environ['CXX'] = 'g++-11'

Options.annotate = True
compiler_directives = {"boundscheck": False, "cdivision": True,
                       "wraparound":False}

# compiler_directives = {"boundscheck": True, "cdivision": False,
#                        "wraparound":True}
ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='paicos',
    ext_modules=cythonize(ext_modules,
                          compiler_directives=compiler_directives))
