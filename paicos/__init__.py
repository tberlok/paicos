# The main classes
from . import util
from .util import root_dir
from . import units


def use_units(use_units):
    """
    pa.use_units(True) turns on paicos quantities globally
    pa.use_units(False) loads in data without applying units
    """
    # from . import util
    units.enabled = use_units


def add_user_function(variable_string, function):
    util.user_functions.update({variable_string: function})


def use_only_user_functions(use_only_user_functions):
    util.use_only_user_functions = use_only_user_functions


def set_numthreads(numthreads):
    util.numthreads = numthreads


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
from .histogram2D import Histogram2D
from .image_reader import ImageReader
# Cython functions
from .cython.openmp_info import simple_reduction, get_openmp_settings
from .cython.get_index_of_region_functions import get_index_of_region
from .cython.get_index_of_region_functions import get_index_of_region_plus_thin_layer
from .cython.get_index_of_region_functions import get_index_of_x_slice_region
from .cython.get_index_of_region_functions import get_index_of_y_slice_region
from .cython.get_index_of_region_functions import get_index_of_z_slice_region
from .cython.sph_projectors import project_image, project_image_omp
from .cython.sph_projectors import project_image2, project_image2_omp
from .cython.get_derived_variables import get_magnitude_of_vector, get_curvature
from .cython.histogram import get_hist_from_weights_and_idigit
from .cython.histogram import get_hist2d_from_weights
from .cython.histogram import find_normalizing_norm_of_2d_hist
