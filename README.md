[![CircleCI](https://dl.circleci.com/status-badge/img/gh/tberlok/paicos/tree/main.svg?style=svg&circle-token=dbdb37aa907d919a167a8ef5ccf197c0d358c300)](https://dl.circleci.com/status-badge/redirect/gh/tberlok/paicos/tree/main)
[![pylint](https://github.com/tberlok/paicos/actions/workflows/pylint.yml/badge.svg)](
https://github.com/tberlok/paicos/actions/workflows/pylint.yml)
[![flake8](https://github.com/tberlok/paicos/actions/workflows/flake8.yml/badge.svg)](
https://github.com/tberlok/paicos/actions/workflows/flake8.yml)
[![Documentation Status](https://readthedocs.org/projects/paicos/badge/?version=latest)](https://paicos.readthedocs.io/en/latest/?badge=latest)


# Paicos

A somewhat bare-bones Python package for making projections and slices of
Arepo simulations. Please note that while Paicos has its visibility set to
public, it is still in beta mode and under active development.

<img src="images/Z24_snap130_wide_projection_notnested.jpg" width="auto">


## Installation on your laptop

> **Warning**
Please do not try to pip install paicos. This is not yet supported and the mess that such an attempt creates will make the instructions below fail.

We clone the repo, pip install the requirements and then compile the code:

```
git clone git@github.com:tberlok/paicos.git
cd paicos
pip install -r requirements.txt
make
```

Assuming this succeeds, you will then need to add the paicos directory to your Python path,
For instance, I have
```
export PYTHONPATH=$PYTHONPATH:/Users/berlok/projects/paicos
```
in my `.bash_profile`.

You can then check that everything works by doing
```
make checks
```

#### Note for installation on MacOs

Paicos requires a compiler with OpenMP support. I have installed gcc-12 via Homebrew and this is currently hardcoded in setup.py. You will have to manually modify setup.py if you do not have this compiler installed.

## Installation for use with Jupyter notebooks on the AIP Newton cluster

First clone the repo onto the AIP newton cluster and then add the path
to your .bash_profile (on the cluster!). I have, for instance,

```
export PYTHONPATH=$PYTHONPATH:/llust21/berlok/paicos
```
You might need to restart singularity in order for this change to the PYTHONPATH to be visible inside the jupyter notebooks.

Now compile the code from inside a notebook using bash magic (replace with
path to your own clone of paicos):

```
%%bash

cd /llust21/berlok/paicos
make clean
make
```

## Using the code

The examples require an Arepo snapshot. You can download one [here](https://www.dropbox.com/sh/xdmqpc72jprtfs7/AADTmM12Zqc4K5--R5OTb4oCa?dl=0) (1 GB Dropbox link).

The main functionality is contained inside some main classes. These are:

- Snapshot (arepo_snap.py)

- Catalog (arepo_catalog.py)

- Slicer (slicer.py)

- Projector (projector.py)

- NestedProjector (nested_projector.py)

- ArepoImage (arepo_image.py)

- PaicosWriter (paicos_writer.py)

- PaicosReader (paicos_readers.py)

Each of these can by run in python, displaying their functionality, e.g.,

```
#### Basic features

# Simple example of loading a snapshot
python3 examples/loading_data_example.py

# Slicing
python3 examples/slicer_example.py

# Projections
python3 examples/projector_example.py
python3 examples/nested_projector_example.py

# Histograms
python3 examples/histogram1D_example.py
python3 examples/histogram2d_example.py

# Loading images
python3 examples/image_reader_example.py

# Creating radial profiles
python3 examples/create_radial_profiles_hdf5file_example.py
python3 examples/read_radial_profile_example.py

# Make a time series
python3 examples/paicos_time_series_example.py

# Select an index of a snapshot
python3 examples/select_subset_of_snap.py

#### Advanced features

# Setting up user-defined functions for obtaining derived variables
python3 examples/user_defined_functions_example.py

# Using a configuration script
python3 examples/using_a_paicos_config.py
python3 examples/example_paicos_config.py

# Saving a reduced snapshot file
python3 examples/save_reduced_file_example.py

# Using aliases
python3 examples/using_aliases_example.py
```

## Tutorial Jupyter notebooks

I have uploaded a few tutorial notebooks in `notebook-tutorials` which
displays the functionality.




