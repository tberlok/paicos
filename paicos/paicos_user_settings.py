import paicos as pa

"""

"""

# Boolean determining whether we use Paicos quantities as a default
pa.use_units(True)

# Settings for automatic calculation of derived variables
pa.use_only_user_functions(False)
pa.print_info_when_deriving_variables(True)

# Number of threads to use in calculations
pa.numthreads(4)

# Info about the openMP setup
pa.give_openMP_warnings(True)

# Whether to load GPU/cuda functionality on startup
pa.load_cuda_functionality_on_startup(True)

# Explicitly set data directory (only needed for pip installations)
data_dir = '/groups/astro/lperrone/paicos/data'
