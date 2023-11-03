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
The `setup.py` in the Paicos directory currently has the following hardcoded
compiler option. 
```
if sys.platform == 'darwin':
    os.environ['CC'] = 'gcc-13'
    os.environ['CXX'] = 'g++-13'
```
If you have installed gcc via homebrew, then you can get your compiler version by running
`brew info gcc` and update `setup.py` accordingly if that does not agree with what is hardcoded.


#### Compiling Paicos inside a notebook
It can sometimes also be useful to compile Paicos inside a Jupyter notebook, i.e.,
by running a notebook cell with the following content:
```
%%bash

cd /path/to/paicos/
make clean
make
```

## Option 2: Installing the development version using pip 
```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt 
python3 -m pip install -e .
make checks
```

## Option 3: Installing directly via pip

This installation does not include the tests and examples and will probably mostly work on Linux machines
```
pip install paicos
```

```
# Check if installation worked and that you can import Paicos 
python -c "import paicos"
```

## Using openmp parallel execution of code

It is sometimes useful to set the environment variable OMP_NUM_THREADS,
which will then be the maximum number of threads that Paicos uses.
```
export OMP_NUM_THREADS=16
```