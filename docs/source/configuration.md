# User configuration

Paicos provides a few ways of configuring its default behaviour. These are described here.

We recommend that you work through notebook 1a before reading on (the notebooks
can be seen on the sidebar on the left).

## Turn units/automatic derivations on/off

Some users might not like the automatic handling of units
and/or the automatic computations of derived quantities.
These features are optional and can be turned off.

You can turn off units by running
```
pa.use_units(False)
```
The reader/writers then simply load/save numpy arrays instead of Paicos quantities.

You can also turn off automatic derivations by running the following code.
```
import paicos as pa
pa.use_only_user_functions(True)
```
This can be used to turn off the library of functions supplied by Paicos
and replace them with your own functions instead (see the "Custom user functions"
section below).

It is worth noting that these options should be modified before loading any snapshots,
i.e. at the top of your analysis scripts or in your `user_settings.py` file (see below).

## Custom user units

dfdf

## Custom user functions

Every research project has its own physical quantities of interest
and people often like to implement this functionality themselves.

Below is an example where the user would like to have Paicos
be abble to automatically compute a derived quantity:
```
import paicos as pa
def TemperaturesTimesMassesSquared(snap, get_depencies=False):
    if get_depencies:
        return ['0_Temperatures', '0_Masses']
    return snap['0_Temperatures'] * snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
```
After setting this up, one would be able to run e.g.

```
snap = pa.Snapshot(root_dir + '/data', 247)

# Automatically compute temperature times masses^2 using the user supplied function
snap['0_TM2']
```

## Setting up user settings

You can save a `user_settings.py` script at in Paicos base directory. We include an example
named ``, which you use to get started. It looks like this

```
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
    return snap['0_TemperaturesTimesMasses'] * snap['0_Masses']


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)

```

## Openmp parallel execution of code

Paicos will check upon startup how many cores are available on your system.


It is sometimes useful to set the environment variable OMP_NUM_THREADS,
which will then be the maximum number of threads that Paicos uses.
```
export OMP_NUM_THREADS=16
```