"""
Paicos is an object-oriented Python package for analysis of (cosmological)
simulations performed with Arepo. Its main feature is that it includes
handling of units, derived variables, loading of data upon request.

The module includes a way of writing and reading data with units.
The code is parallel with an OpenMP Cython implementation.
"""

__version__ = "0.1.4"
__author__ = 'Thomas Berlok'
__credits__ = 'Leibniz-Institute for Astrophysics Potsdam (AIP)'

# Dependencies
import os
import numpy
import scipy
import h5py
import astropy

# Settings and utility functions
from . import util
from . import settings
from .util import root_dir

# HDF5 file readers
from .readers.arepo_snap import Snapshot
from .readers.arepo_catalog import Catalog
from .readers.paicos_readers import PaicosReader, ImageReader, Histogram2DReader

# HDF5 file writers
from .writers.paicos_writer import PaicosWriter, PaicosTimeSeriesWriter
from .writers.arepo_image import ArepoImage

# Image creators
from .image_creators.image_creator import ImageCreator
from .image_creators.projector import Projector
from .image_creators.nested_projector import NestedProjector
from .image_creators.tree_projector import TreeProjector
from .image_creators.slicer import Slicer

# Histograms
from .histograms.histogram import Histogram
from .histograms.histogram2D import Histogram2D

# Derived variables
from .derived_variables import derived_variables

# Cython functions
from . import cython

# pylint: disable=W0621


def use_units(use_units):
    """
    pa.use_units(True) turns on paicos quantities globally
    pa.use_units(False) loads in data without applying units.

    The status can be seen in pa.settings.use_units
    """
    settings.use_units = use_units


def add_user_function(variable_string, function):
    """
    This functions allows to enable user functions for obtaining
    deriving variables. An example could be the following:

    def TemperaturesTimesMassesSquared(snap, get_depencies=False):
        if get_depencies:
            return ['0_Temperatures', '0_Masses']
        return snap['0_Temperatures'] * snap['0_Masses']**2


    pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)

    """
    derived_variables.user_functions.update({variable_string: function})


def use_only_user_functions(use_only_user_functions):
    """
    Passing 'True' to this functions disables automatic derivation of
    variables except for the ones you have added via 'add_user_function'.
    """
    settings.use_only_user_functions = use_only_user_functions


def add_user_unit(field, blockname, unit):
    """
    This function adds user units.

    Parameters:

    field (string): possible values are specified in
                    unit_specifications.pos_fields, currently

                    pos_fields = ['default', 'voronoi_cells', 'dark_matter',
                    'stars', 'black_holes', 'groups', 'subhalos']

    blockname (string): The blockname that you would like to enable units for.

    unit (string): The unit that the blockname should have.
                                   String inputs are preferred, e.g.,

                         'arepo_mass arepo_length small_a^2 small_h^(-3/2)'

                   The available arepo units are

                   arepo_length (often 1 kpc)
                   arepo_mass (often 10^10 Msun)
                   arepo_velocity (often 1 km/s)
                   arepo_time
                   arepo_energy
                   arepo_pressure
                   arepo_density

                   Please note that these only become available once you
                   actually load a hdf5 file. This is the reason that you can
                   pass the string only.
    """
    if field not in util.user_unit_dict:
        raise RuntimeError(f'unknown field: {field}')

    util.user_unit_dict[field][blockname] = unit


def numthreads(numthreads):
    """
    Pass the number of threads you would like to use for parallel execution.

    numthreads (int): e.g. 16
    """
    util.check_if_omp_has_issues(False)

    if numthreads > settings.max_threads:
        print(f'Your machine only has {settings.max_threads} available threads')

    settings.numthreads = min(numthreads, settings.max_threads)
    if settings.openMP_has_issues:
        settings.numthreads_reduction = 1
    else:
        settings.numthreads_reduction = settings.numthreads


def print_info_when_deriving_variables(option):
    """
    Input: a boolean controlling whether to provide info to terminal.
    """
    settings.print_info_when_deriving_variables = option


def give_openMP_warnings(option):
    """
    Turns of warnings that will otherwise appear on a mac computer.
    """
    settings.give_openMP_warnings = option


def set_aliases(aliases):
    """
    Assign a list of aliases for use as keys in snapshot objects.

    input (dictionary): e.g.

    aliases = {'0_Density': 'dens',
           '0_Temperatures': 'T',
           '0_MeanMolecularWeight': 'mu'}
    """
    msg = 'Your user-set aliases seem to not be unique'
    assert len(set(aliases.values())) == len(aliases.values()), msg

    inverse = {aliases[key]: key for key in aliases.keys()}
    settings.aliases = aliases
    settings.inverse_aliases = inverse
    settings.use_aliases = True


def user_settings_exists():
    """
    Checks if user settings exist in the root directory of Paicos.
    """
    if os.path.exists(root_dir + 'user_settings.py'):
        return True
    return False


if user_settings_exists():
    # pylint: disable=E0401
    import user_settings

# Do this at start up
util.check_if_omp_has_issues()
