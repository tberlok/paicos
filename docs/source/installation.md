# How to install

## Creating an environment

We recommend installing conda if you don’t already have it. Follow instructions at <https://github.com/conda-forge/miniforge/#download>

Once you have that working you can create an environment and activate it
```
conda create -q -n paicos python=3.10
conda activate paicos
```

Below we provide three differens ways of getting Paicos up and running,

## Option 1: Compile the code and add its path to your PYTHONPATH 
```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt 
make
make checks
```

Add the directory to your PYTHONPATH, e.g., I have
```
export PYTHONPATH=$PYTHONPATH:/Users/berlok/projects/paicos
```
in my `.bash_profile`.

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

## Option 2: Installing directly via pip

This is probably the simplest way of installing Paicos, i.e. you run
```
pip install paicos
```
in a terminal.

Note however, that this installation does not include the tests, examples
and example data.

On MacOs you might have to do something like (see details above)
```
CC=gcc-13 pip install paicos
```
If the installation succeeded then you can proceed:
```
# Check if installation worked and that you can import Paicos 
python -c "import paicos"
```

## Option 3: Installing the development version using pip 
```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt 
python3 -m pip install -e .
make checks
```

## GPU/CUDA requirements

The visualization routines that run on GPU require installing CuPy (a drop-in replacement
for NumPy that runs on the GPU) and Numba CUDA (just-in-time compilation of kernel
and device functions on the GPU). These packages only work on CUDA-enabled GPUs,
which means that you need a recent Nvidia GPU.

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


