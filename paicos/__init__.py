# The main classes
from . import util
from . import settings
from .util import root_dir
from .arepo_image import ArepoImage, ImageCreator
from .arepo_snap import Snapshot
from .arepo_catalog import Catalog
from .projector import Projector
from .projector2 import Projector2
from .nested_projector import NestedProjector
from .slicer import Slicer
from .arepo_converter import ArepoConverter
from .radial_profiles import RadialProfiles
from .histogram import Histogram
from .histogram2D import Histogram2D, Histogram2DReader
from .image_reader import ImageReader

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
