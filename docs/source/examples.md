# Running the examples and notebooks

We provide a number of examples in the subdirectories `examples/` and `ipython-notebooks`.
We recommend that you start by working your way through the notebooks, starting with
notebook 1a.

## Downloading sample data

The examples require an Arepo snapshot. You can download one [here](https://www.dropbox.com/sh/xdmqpc72jprtfs7/AADTmM12Zqc4K5--R5OTb4oCa?dl=0) (1 GB Dropbox link). Alternatively, open a terminal at
the root directory of the Paicos repo, then run
```
wget -O data/fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/BmILDaDPPz
wget -O data/snap_247.hdf5 https://sid.erda.dk/share_redirect/G4pUGFJUpq
```

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