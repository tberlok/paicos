"""
Paicos is an object-oriented Python package for analysis of (cosmological)
simulations performed with Arepo. Its main feature is that it includes
handling of units, derived variables, loading of data upon request.

The module includes a way of writing and reading data with units.
The code is parallel with an OpenMP Cython implementation
and a CUDA GPU implementation for visualization.
"""

__version__ = "0.1.16"
__author__ = 'Thomas Berlok'
__credits__ = 'Niels Bohr Institute, University of Copenhagen'

# Dependencies
import os
import numpy
import scipy
import h5py
import astropy

# Settings and utility functions
from . import util
from . import settings
from . import units

# One folder up from __init__ (i.e. repo directory or installation directory)
from .util import root_dir

# HDF5 file readers
from .readers.arepo_snap import Snapshot
from .readers.arepo_catalog import Catalog
from .readers.paicos_readers import PaicosReader, ImageReader, Histogram2DReader
from .readers.generic_snap import GenericSnapshot


# HDF5 file writers
from .writers.paicos_writer import PaicosWriter, PaicosTimeSeriesWriter
from .writers.arepo_image import ImageWriter, ArepoImage

# Image creators
from .image_creators.image_creator import ImageCreator
from .image_creators.projector import Projector
from .image_creators.nested_projector import NestedProjector
from .image_creators.tree_projector import TreeProjector
from .image_creators.ray_projector import RayProjector
from .image_creators.slicer import Slicer
from .image_creators.image_creator_actions import Actions

# Histograms
from .histograms.histogram import Histogram
from .histograms.histogram2D import Histogram2D

# Derived variables
from .derived_variables import derived_variables

# Orientation class
from .orientation import Orientation

# Cython functions
from . import cython

# pylint: disable=W0621

# The place where __init__.py (this file) is located
code_dir = os.path.dirname(os.path.abspath(__file__))
# This is the user home_dir
home_dir = os.path.expanduser('~')


def use_units(use_units):
    """
    Parameters
    ----------

    use_units : bool
        True for enabling units globally and False for disabling
        them globally.
        Note that this will not modify snapshots that have
        already been loaded.

    Examples
    --------

    An example::

        import paicos as pa
        print('Using units:', pa.settings.use_units)

        # Turn on paicos quantities globally
        pa.use_units(True)
        print('Using units:', pa.settings.use_units)

        # Loads in data without applying units.
        pa.use_units(False)
        print('Using units:', pa.settings.use_units)

    """
    settings.use_units = use_units


def add_user_function(variable_string, function):
    """
    This functions allows to enable user functions for obtaining
    deriving variables. An example could be the following::

        def TemperaturesTimesMassesSquared(snap, get_depencies=False):
            if get_depencies:
                return ['0_Temperatures', '0_Masses']
            return snap['0_Temperatures'] * snap['0_Masses']**2


        pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)

    """
    from inspect import signature
    if ' ' in variable_string:
        raise RuntimeError(f'{variable_string} contains space(s)!')

    sig = signature(function)
    if len(sig.parameters) == 2:
        dependencies = function(None, True)
        assert isinstance(dependencies, list)
        err_msg = 'Problem with list of dependencies!'
        for item in dependencies:
            assert isinstance(item, str), err_msg
            assert ' ' not in item, err_msg

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

    Parameters
    ----------

    field : string
        possible values are specified in
        unit_specifications.pos_fields, currently

        pos_fields = ['default', 'voronoi_cells', 'dark_matter',
        'stars', 'black_holes', 'groups', 'subhalos']

    blockname : string
        The blockname that you would like to enable units for.

    unit : string
        The unit that the blockname should have.
        String inputs are preferred, e.g., ``arepo_mass arepo_length small_a^2 small_h^(-3/2)``.

        The available arepo units are:

        ``arepo_length`` (often 1 kpc)

        ``arepo_mass`` (often 10^10 Msun)

        ``arepo_velocity`` (often 1 km/s)

        ``arepo_time``

        ``arepo_energy``

        ``arepo_pressure``

        ``arepo_density``

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
    util._check_if_omp_has_issues(False)

    if numthreads > settings.max_threads:
        print(f'Your machine only has {settings.max_threads} available threads')

    settings.numthreads = min(numthreads, settings.max_threads)
    if settings.openMP_has_issues:
        settings.numthreads_reduction = 1
    else:
        settings.numthreads_reduction = settings.numthreads

    try:
        from numba import set_num_threads
        from .misc import suppress_omp_nested_warning

        with suppress_omp_nested_warning():
            set_num_threads(settings.numthreads)
    except Exception as e:
        print(e)
        import warnings
        warnings.warn("Something went wrong. Perhaps numba is not installed?")


def print_info_when_deriving_variables(option):
    """
    Input: a boolean controlling whether to provide info to terminal.
    """
    settings.print_info_when_deriving_variables = option


def give_openMP_warnings(option):
    """
    Turns of warnings that might otherwise appear on a mac computer.
    """
    settings.give_openMP_warnings = option


def load_cuda_functionality_on_startup(option):
    """
    Turns on/off whether to import GPU/cuda on startup.
    """
    settings.load_cuda_functionality_on_startup = option


def set_aliases(aliases):
    """
    Assign a list of aliases for use as keys in snapshot objects.

    Parameters
    ----------
        aliases : dict
            A dictionary containing the mapping of variable names
            that you would like to have enabled, e.g.::

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


def import_user_settings():
    """
    Import user settings if they exist in the root directory of Paicos
    or as a hidden file at the users home directory.

    :meta private:
    """
    data_dir = None
    if os.path.exists(root_dir + 'data/'):
        data_dir = root_dir + 'data/'
    if os.path.exists(code_dir + '/paicos_user_settings.py'):
        from . import paicos_user_settings
        if hasattr(paicos_user_settings, 'data_dir'):
            data_dir = paicos_user_settings.data_dir
    if os.path.exists(home_dir + '/.paicos_user_settings.py'):
        filepath = home_dir + '/.paicos_user_settings.py'
        import importlib
        spec = importlib.util.spec_from_file_location('paicos_settings',
                                                      location=filepath)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        if hasattr(foo, 'data_dir'):
            data_dir = foo.data_dir

    if data_dir is not None:
        if data_dir[-1] != '/':
            data_dir += '/'

    return data_dir


data_dir = import_user_settings()


# Import of GPU functionality only if
# a simple test with cupy and numba works.
def gpu_init(gpu_num=0):
    """
    Calling this function initializes the GPU parts of the code.

    You can set settings.load_cuda_functionality_on_startup = True,
    to do this automatically on startup, but you should be aware
    that loading cupy and numba can be a bit time-consuming,
    so that this significantly affects the import time.

    Parameters:
    ----------

        gpu_num (int): The GPU that you want to use for computations,
                       i.e., we call cupy.cuda.Device(gpu_num).use()
    """

    try:
        import cupy as cp
        from numba import cuda

        if gpu_num != 0:
            cp.cuda.Device(gpu_num).use()

        @cuda.jit
        def my_kernel(io_array):
            pos = cuda.grid(1)
            if pos < io_array.size:
                io_array[pos] *= 2

        data = cp.ones(10**6)
        threadsperblock = 256
        blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
        my_kernel[blockspergrid, threadsperblock](data)

        del data

        # Test above worked, do the imports
        from .image_creators.gpu_sph_projector import GpuSphProjector
        from .image_creators.gpu_ray_projector import GpuRayProjector
        import paicos
        paicos.GpuSphProjector = GpuSphProjector
        paicos.GpuRayProjector = GpuRayProjector
    except Exception as e:
        import warnings
        print(e)
        err_msg = ('\nPaicos: The simple cuda example using cupy and numba failed '
                   'with the error above. Please check the official documentation for '
                   'cupy and numba for installation procedure. Note that you need '
                   'a GPU that supports CUDA.\n')
        warnings.warn(err_msg)


if settings.load_cuda_functionality_on_startup:
    gpu_init()


# Do this at start up
util._check_if_omp_has_issues()
