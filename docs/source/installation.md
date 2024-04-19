# How to install

## Creating an environment

We recommend installing conda if you don’t already have it. Follow instructions at <https://github.com/conda-forge/miniforge/#download>

Once you have that working, you can test that you can create an environment and activate it.

Below we then provide a few different ways of getting Paicos up and running inside your conda environment:

## Option 1: Installing directly via PyPi or conda-forge

This is probably the simplest way of installing Paicos. Note however, that
these installations do not include the tests, examples and example data.
You can always download these separately or simply peruse the examples documented online.

#### Installation using the conda-forge distribution

Copy-paste below commands into a terminal:
```
conda create -q -n paicos-conda python=3.11 --yes
conda activate paicos-conda
conda install paicos --yes
# conda install pytest pytest-order cython --yes # Uncomment this line to also install developer dependencies
```

#### Installation using the PyPi distribution 

Copy-paste below commands into a terminal:
```
conda create -q -n paicos-pypi python=3.10 --yes
conda activate paicos-pypi
pip install paicos[dev]
```

On MacOs you might have to do something like this (see details below in "Compiling on MacOs")
```
CC=gcc-13 pip install paicos[dev]
```

#### Check that the installation succeeded

If the installation succeeded then you can proceed:
```
# Check if installation worked and that you can import Paicos 
python -c "import paicos"
```

## Option 2 (Developer installation): Compile the code and add its path to your PYTHONPATH 

You should use this option if you intend to make changes to Paicos ([this would be very welcome, see here for details on making contributions])(https://github.com/tberlok/paicos/blob/main/.github/CONTRIBUTING.md)
```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt
pip install -r dev_requirements.txt
make
```

Add the directory to your PYTHONPATH, e.g., I have
```
export PYTHONPATH=$PYTHONPATH:/Users/berlok/projects/paicos
```
in my `.bash_profile`. Finally, run the tests
```
make checks
```

#### Compiling on MacOs

You will need an openmp-enabled compiler. On MacOs, you can install gcc via homebrew.
You can then either modify `setup.py` with your
compiler option, .e.g.,
```
if sys.platform == 'darwin':
    os.environ['CC'] = 'gcc-13'
    os.environ['CXX'] = 'g++-13'
```
or by running `CC=gcc-13 make` instead of simply `make`.

If you have installed gcc via homebrew, then you can get your compiler version by running
`brew info gcc`, which you can use to modify the instructions above accordingly.


#### Compiling Paicos inside a notebook
It can sometimes also be useful to compile Paicos inside a Jupyter notebook, i.e.,
by running a notebook cell with the following content:
```
%%bash

cd /path/to/paicos/
make clean
make
```

<!---
## Option 3: Installing the development version using pip 
```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt
pip install -r dev_requirements.txt
python3 -m pip install -e .
make checks
```
-->

## GPU/CUDA requirements

The visualization routines that run on GPU require installing CuPy (a drop-in replacement
for NumPy that runs on the GPU) and Numba CUDA (just-in-time compilation of kernel
and device functions on the GPU). These packages only work on CUDA-enabled GPUs,
which means that you need a recent Nvidia GPU. An Nvidia GPU with good FP64 performance
is desirable.

These packages (CuPy and Numba) are not automatically included in Paicos.
Up-to-date instructions for installing them can be found at:

- CuPy: https://docs.cupy.dev/en/stable/install.html

- Numba: https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus

At the time of writing, we have had success installing for CUDA version
11.2 using

```
pip install numba
pip install cupy-cuda112
```
and then setting the path to the CUDA installation in .bashrc as e.g.
(substitute with the path to the CUDA installation on your system)
```
export CUDA_HOME=/software/astro/cuda/11.2 # numba
export CUDA_PATH=/software/astro/cuda/11.2 # cupy
```

Finally, you need to add
```
# Whether to load GPU/cuda functionality on startup
pa.load_cuda_functionality_on_startup(True)
```
to your `paicos_user_settings.py` (see details under 'User configuration').


