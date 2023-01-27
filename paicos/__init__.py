# The main classes
from . import util
from . import settings
from .util import root_dir
from .arepo_image import ArepoImage, ImageCreator
from .arepo_snap import Snapshot
from .arepo_catalog import Catalog
from .projector import Projector
from .nested_projector import NestedProjector
from .slicer import Slicer
from .arepo_converter import ArepoConverter
from .paicos_writer import PaicosWriter
from .paicos_readers import PaicosReader, ImageReader, Histogram2DReader
from .histogram import Histogram
from .histogram2D import Histogram2D

# Cython functions
from . import cython


def use_units(use_units):
    """
    pa.use_units(True) turns on paicos quantities globally
    pa.use_units(False) loads in data without applying units.

    The status can be seen in pa.settings.use_units
    """
    settings.use_units = use_units


def add_user_function(variable_string, function):
    util.user_functions.update({variable_string: function})


def use_only_user_functions(use_only_user_functions):
    settings.use_only_user_functions = use_only_user_functions


def numthreads(numthreads):
    settings.numthreads = numthreads


def print_info_when_deriving_variables(option):
    """
    Input: a boolean controlling whether to provide info to terminal.
    """
    settings.print_info_when_deriving_variables = option


def give_openMP_warnings(option):
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
    import os
    if os.path.exists(root_dir + 'user_settings.py'):
        return True
    else:
        return False


if user_settings_exists():
    import user_settings
