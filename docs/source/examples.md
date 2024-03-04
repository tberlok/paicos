# Running the examples and notebooks

We provide a number of examples in the subdirectories `examples/` and `notebook-tutorials`.
We recommend that you start by working your way through the notebooks, starting with
notebook 1a.

## Downloading sample data

The examples require an Arepo snapshot. You can download one [here](https://sid.erda.dk/sharelink/DkVXfdoxIM). Alternatively, open a terminal at
the root directory of the Paicos repo, then run
```
wget -O data/fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/BmILDaDPPz
wget -O data/snap_247.hdf5 https://sid.erda.dk/share_redirect/G4pUGFJUpq
```

If you have done a pip-installation, then you will need to tell Paicos where the data is on your system.
This is done by adding something like this
```
# Explicitly set data directory (only needed for pip installations)
data_dir = '/Users/berlok/projects/paicos/data/'
```
to your `paicos_user_settings.py` (see the user configuration tab).

## Overview of the main Paicos classes
The notebooks and python scripts illustrate typical use cases of the following Paicos classes:

- Snapshot (arepo_snap.py)

- Catalog (arepo_catalog.py)

- Slicer (slicer.py)

- Projector (projector.py)

- NestedProjector (nested_projector.py)

- TreeProjector (tree_projector.py)

- ArepoImage (arepo_image.py)

- PaicosWriter (paicos_writer.py)

- PaicosReader (paicos_readers.py)

## Main GPU classes

The main GPU classes are

- GpuSphProjector (gpu_sph_projector.py)
- GpuRayProjector (gpu_ray_projector.py)
