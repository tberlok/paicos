# Paicos

A somewhat bare-bones python package for making projections and slices of
Arepo simulations.

<img src="images/Z24_snap130_wide_projection_notnested.jpg" width="700">


### Installation on your laptop

Clone the repo, then:

```
pip install -r requirements.txt
```
followed by
```
make
```

### Installation for use with Jupyter notebooks on the AIP Newton cluster

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

The main functionality is contained inside 6 classes. These are:

- Projector (projector.py)

- NestedProjector (nested_projector.py)

- Slicer (slicer.py)

- RadialProfiles (radial_profiles.py)

- ArepoImage (arepo_image.py)

- ArepoConverter (arepo_converter.py)

Each of these can by run in python, displaying their functionality, e.g.,

```
python paicos/slicer.py
```

### Example Jupyter notebooks

I will upload some of these soon, please ask again if 

