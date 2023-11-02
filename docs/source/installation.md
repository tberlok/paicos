# How to install

## Creating an environment

We recommend installing conda if you don’t already have it. Follow instructions at <https://github.com/conda-forge/miniforge/#download>

Once you have that working you can create an environment and activate it
```
conda create -q -n paicos python=3.10
conda activate paicos
```

Below we provide a few different ways of getting Paicos up and running,

You will need an openmp-enabled compiler. On MacOs, you can install gcc via homebrew.

## 1. Compile the code and add its path to your PYTHONPATH 
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt 
make
make checks

Add the directory to your PYTHONPATH, e.g.


This method can also be useful for compiling Paicos inside a Jupyter notebook, i.e.,
by running a cell with the following content:

## 2. Installing the development version using pip 
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt 
python3 -m pip install -e .
make checks

## Downloading the example data files 
wget -O data/fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/BmILDaDPPz
wget -O data/snap_247.hdf5 https://sid.erda.dk/share_redirect/G4pUGFJUpq

## 3. Installing directly via pip

This installation does not include the tests and examples and will probably mostly work on Linux machines

pip install paicos

Check if installation worked and you can import Paicos 
python -c “import paicos”

## Using openmp parallel execution of code

It is sometimes useful to set the environment variable OMP_NUM_THREADS,
which will then be the maximum number of threads that Paicos uses.
export OMP_NUM_THREADS=16