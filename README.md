# Paicos

A somewhat bare-bones python package for making projections and slices of
Arepo simulations.

<img src="images/Z24_snap130_wide_projection_notnested.jpg" width="auto">


## Installation on your laptop

Clone the repo, then:

```
pip install -r requirements.txt
```
followed by
```
make
```

## Installation for use with Jupyter notebooks on the AIP Newton cluster

Compile the code from inside a notebook using bash magic (replace with
path to your own clone of paicos):

```
%%bash

cd /llust21/berlok/paicos
make clean
make
```

## Using the code

The examples require an Arepo snapshot. You can download one [here](https://www.dropbox.com/sh/xdmqpc72jprtfs7/AADTmM12Zqc4K5--R5OTb4oCa?dl=0) (1 GB Dropbox link).

The main functionality is contained inside 8 classes. These are:

- Snapshot (arepo_snap.py)

- Catalog (arepo_catalog.py)

- Slicer (slicer.py)

- Projector (projector.py)

- NestedProjector (nested_projector.py)

- RadialProfiles (radial_profiles.py)

- ArepoImage (arepo_image.py)

- ArepoConverter (arepo_converter.py)

Each of these can by run in python, displaying their functionality, e.g.,

```
# Simple example of loading a snapshot
python examples/loading_data_example.py

# Slicing
python examples/slicer_example.py

# Projections
python examples/projector_example.py
python examples/nested_projector_example.py


# Histograms
python examples/histogram1D_example.py
python examples/histogram2d_example.py

# Loading images
python examples/image_reader_example.py

# Creating radial profiles
python examples/radial_profiles_example.py

# Setting up user-defined functions for obtaining derived variables
python examples/user_defined_functions_example.py

# Using a configuration script
python examples/using_a_paicos_config.py
python examples/example_paicos_config.py

# Select an index of a snapshot
python examples/select_subset_of_snap.py
```

## Tutorial Jupyter notebooks

I have uploaded a tutorial notebook `tutorials/paicos_examples.ipynb` which
displays the functionality on one of the bigger simulations on the cluster.

