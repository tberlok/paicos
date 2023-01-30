import paicos as pa

"""
Set up your own default settings by renaming this file as user_settings.py.

Here we are overriding the defaults set in settings.py, so you only need
to add things you want to change.
"""

# Boolean determining whether we use Paicos quantities as a default
pa.use_units(True)

# Settings for automatic calculation of derived variables
pa.use_only_user_functions(False)
pa.print_info_when_deriving_variables(False)

# Number of threads to use in calculations
pa.numthreads(8)

# Info about the openMP setup
pa.give_openMP_warnings(False)


# Examples of adding user-defined functions
def TemperaturesTimesMassesSquared(snap, get_dependencies=False):
    if get_dependencies:
        return ['0_TemperaturesTimesMasses', '0_Masses']
    return snap['0_TemperaturesTimesMasses']*snap['0_Masses']


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
