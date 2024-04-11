import paicos as pa

"""
Set up your own default settings by renaming this file as paicos_user_settings.py
and saving it at the directory found at: pa.code_dir
or as a hidden file in your home directory (pa.home_dir), e.g.
on my (Thomas Berlok) laptop '/Users/berlok/.paicos_user_settings.py'

Here we are overriding the defaults set in settings.py, so you only need
to add things you want to change.
"""

# Boolean determining whether we use Paicos quantities as a default
pa.use_units(True)

# Settings for automatic calculation of derived variables
pa.use_only_user_functions(False)
pa.print_info_when_deriving_variables(False)

# Number of threads to use in calculations
pa.numthreads(16)

# Info about the openMP setup
pa.give_openMP_warnings(False)

# Whether to load GPU/cuda functionality on startup
pa.load_cuda_functionality_on_startup(False)

# Explicitly set data directory (only needed for pip installations)
data_dir = '/Users/berlok/projects/paicos/data/'


# Examples of adding user-defined functions
def TemperaturesTimesMassesSquared(snap, get_dependencies=False):
    if get_dependencies:
        return ['0_TemperaturesTimesMasses', '0_Masses']
    return snap['0_TemperaturesTimesMasses'] * snap['0_Masses']


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
