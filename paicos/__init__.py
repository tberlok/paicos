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
    util.use_paicos_quantities = use_units


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
from .image_reader import ImageReader
# Some useful functions
from .derived_variables import get_variable
# Cython functions
from .cython.openmp_info import simple_reduction, print_openmp_settings
from .cython.get_index_of_region_functions import get_index_of_region
from .cython.get_index_of_region_functions import get_index_of_region_plus_thin_layer
from .cython.get_index_of_region_functions import get_index_of_x_slice_region
from .cython.get_index_of_region_functions import get_index_of_y_slice_region
from .cython.get_index_of_region_functions import get_index_of_z_slice_region
from .cython.sph_projectors import project_image, project_image_omp
from .cython.sph_projectors import project_image2, project_image2_omp
from .cython.get_derived_variables import get_magnitude_of_vector, get_curvature
from .cython.histogram import get_hist_from_weights_and_idigit
