# Boolean determining whether we use Paicos quantities as a default
use_units = True

# Number of threads to use in calculations
numthreads = 8

# Settings for automatic calculation of derived variables
use_only_user_functions = False
print_info_when_deriving_variables = True

# Load all fields as double precision arrays (FALSE is currently not supported
# by the Cython routines, so True is strongly recommended)
double_precision = True

# OpenMP info
give_openMP_warnings = True

# Whether to use aliases, e.g. dens instead of 0_Density
use_aliases = False

# Strictly enforce units
strict_units = True

# Enable spherical coordinates
spherical_coords = False

# Enable cylindrical coordinates
cyl_coords = False

# Whether to load GPU/cuda functionality on startup
load_cuda_functionality_on_startup = False
