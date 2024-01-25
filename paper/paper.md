---
title: 'Paicos: A Python package for analysis of (cosmological) simulations performed with Arepo'
tags:
  - Python
  - astronomy
authors:
  - name: Thomas Berlok
    orcid: 0000-0003-0466-603X
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Léna Jlassi
    affiliation: 2
  - name: Ewald Puchwein
    affiliation: 2
  - name: Troels Haugbølle
    affiliation: 1
affiliations:
 - name: Niels Bohr Institute, University of Copenhagen, Denmark
   index: 1
 - name: Leibniz-Institut für Astrophysik Potsdam, Germany
   index: 2
date: 31 January 2024
bibliography: paper.bib

---

# Summary

Cosmological simulations evolve dark matter and baryons subject to
gravitational and hydrodynamic forces [@Vogelsberger2020]. The simulations
start at high redshift and capture hierarchical structure formation where
small structures form first and later assemble to larger structures
[@Springel2005]. The Arepo code is a versatile finite-volume code which can
solve the magnetohydrodynamic equations on an unstructured Voronoi mesh in a
cosmologically comoving frame [@Springel2010; @Weinberger2020].

Here we present Paicos, a new object-oriented Python package for analyzing
simulations performed with Arepo. Paicos strives to reduce the learning curve
for students and researchers getting started with Arepo simulations. As such,
Paicos includes many examples in the form of Python scripts and Jupyter
notebooks [@Kluyver2016] as well as an online documentation describing
the installation procedure and recommended first steps.

Paicos' main features are automatic handling of cosmological
and physical units, computation of
derived variables, 2D visualization (slices and projections),
1D and 2D histograms, and easy
saving and loading of derived data including units and all the relevant
metadata.

Paicos relies heavily on well-established open source software libraries such
as NumPy [@numpy; @Harris2020], Scipy [@Virtanen2020], h5py [@h5py], Cython
[@Behnel2011] and astropy [@astropy] and contains features for interactive
data analysis inside Ipython terminals [@Perez2007] and Jupyter notebooks
[@Kluyver2016], i.e., tab completion of data keywords and Latex rendering of
data units. Paicos also contains a number of tests that are automated using
pytest [@pytest7.4], CircleCi and GitHub workflows.

# Statement of need

The Arepo code stores its data output as HDF5 files, which can easily be loaded
as NumPy arrays using h5py. However, data visualization of the unstructured mesh
used in Arepo is non-trivial and keeping track of the units used in the data
outputs can also be a tedious task. The general purpose visualization and data analysis
software package yt[^yt] [@yt-Turk] and the visualization
package py-sphviewer [@py-sphviewer] are both able to perform visualizations
of Arepo simulations. Paicos provides an alternative, which is
specifically tailored to Arepo simulations.
It is in this regard
similar to swiftsimio [@Borrow2020], which was developed specifically
for SWIFT simulations [@Schaller2016].

We have developed Paicos because we identified a need for an analysis code
that simultaneously fulfills the following requirements: 1) is specifically
written for analysis of Arepo simulations discretized on a Voronoi mesh 2)
provides safeguards against errors related to conversions of cosmological and
physical units 3) facilitates working with large data sets by supporting the
saving and loading of reduced/derived data including
units and metadata
4) contains enough functionality to be useful for
practical research tasks while still being light-weight and well-documented
enough that it can be installed, used and understood by a junior
researcher with little or no assistance.

# Overview of key Paicos features

The key Paicos features, which were implemented to fulfill the above
requirements, are as follows:

- Functionality for reading cell/particle and group/subhalo
  catalog data as saved by Arepo in the HDF5 format.
- Unit-handling via astropy with additional support for the $a$ and $h$
  factors that are used in cosmological simulations.[^comoving]
- Functionality for automatically obtaining derived variables from the
  variables present in the Arepo snapshot (e.g. the cell volume, temperature,
  or magnetic field strength).
  Within Jupyter notebooks or an
  Ipython terminal: Tab completion to show which variables are available for
  a given data group (either by directly loading from the HDF5-file or by
  automatically calculating a derived variable).
- Functionality for saving and loading data including cosmological and
  physical units in HDF5-files that automatically include all metadata from
  the original Arepo snapshot (i.e. the Header, Config and Param groups).
- Functionality for creating slices and projections and saving them in a
  format that additionally includes information about the size, center and
  orientation of the image.
- Functionality for creating 1D and 2D histograms (e.g. radial profiles and
  $\rho$-$T$ phase-space plots) of large data sets (using OpenMP-enabled
  Cython).
- Functionality for user customization, e.g. adding new units or functions for
  computing derived variables.
- Functionality for selecting only a part of a snapshot for analysis and for
  storing this selection as a new reduced snapshot.

Finally, we also make public GPU implementations of projection
functionalities, which are much faster than the OpenMP-parallel CPU
implementations described above.
The GPU implementations include a mass-conserving SPH-like projection and a
ray tracing implementation using a bounding volume hierarchy (BVH) in the
form of a binary radix tree for nearest neighbor searches. These GPU implementations use
Numba [@Numba2015] and CuPy [@cupy_learningsys2017] and require that the user
has a CUDA-enabled GPU. Our binary tree implementation follows the
the GPU-optimized tree-construction algorithm described in [@Karras2012] and
the implementation of it found in the publicly available Cornerstone
Octree GPU-library [@Keller2023]. Using our SPH-like
implementation on an [Nvidia A100 GPU](https://www.nvidia.com/en-us/data-center/a100/),
it takes 0.15 seconds to
project 93 mio. particles onto an image plane with $2048^2$ pixels. This
speed enables interactive data exploration. We provide an example IPython
widget illustrating this feature. Finally, we note that the returned images
include physical units and can be used for scientific analysis.

Paicos is hosted on GitHub at https://github.com/tberlok/paicos. We strongly
encourage contributions to Paicos by opening issues and submitting pull requests.

[^yt]: See http://yt-project.org/.

[^comoving]: See e.g. @Pakmor2013 and @Berlok2022 for detailed discussions and
a derivation of the comoving magnetohydrodynamic equations used in
cosmological simulations including magnetic fields.


# Acknowledgements

We are grateful to Christoph Pfrommer for support and advice. We thank Rüdiger
Pakmor, Rosie Talbot and Timon Thomas for useful discussions as well as
Matthias Weber, Lorenzo Maria Perrone, Arne Trabert and Joseph Whittingham
for beta-testing Paicos. We are grateful to Volker Springel for making the
Arepo code available and to the main developers of arepo-snap-util
(Federico Marinacci and Rüdiger Pakmor), a non-public code which we have used
for comparison of projections and slices. TB gratefully acknowledges funding
from the European Union’s Horizon Europe research and innovation programme
under the Marie Skłodowska-Curie grant agreement No 101106080. LJ acknowledges
support by the German Science Foundation (DFG) under grant "DFG Research Unit FOR
5195 – Relativistic Jets in Active Galaxies".
LJ and TB acknowledge support by the European Research Council under ERC-AdG grant
PICOGAL-101019746. The authors gratefully acknowledge the Gauss Centre for
Supercomputing e.V. (www.gauss-centre.eu) for funding this project by
providing computing time on the GCS Supercomputer SuperMUC at Leibniz
Supercomputing Centre (www.lrz.de). The Tycho supercomputer hosted at the
SCIENCE HPC center at the University of Copenhagen was used for supporting
this work.

# References