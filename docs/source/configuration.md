# User configuration

Paicos provides a few ways of configuring its default behaviour. These are described here.

We recommend that you work through notebook 1 before reading on (the notebooks
can be seen on the sidebar on the left).

## Turn units/automatic derivations on/off

You might not like the automatic handling of units
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
This can be used to turn off the library of functions supplied by Paicos.
You can then replace them with your own functions instead (see the "Custom user functions"
section below).

It is worth noting that these options should be modified before loading any snapshots,
i.e. at the top of your analysis scripts or in your `paicos_user_settings.py` file (see below).

## Custom user units

If you are using automatic handling of units, you need to set your own units if you use a new output
option in Arepo or if your simulations include some non-standard physics options.

```
# Define a unitless tracer
pa.add_user_unit('voronoi_cells', 'JetTracer', '')
```

This lets Paicos know that your Arepo snapshots contain a block
name JetTracer in the gas cells. You can get the full documentation
by running
```
import paicos as pa
pa.add_user_unit?
```

## Custom user functions

Every research project has its own physical quantities of interest
and people often like to implement this functionality themselves.

Below is an example where the user would like to have Paicos
be able to automatically compute a derived quantity:
```
import paicos as pa
def TemperaturesTimesMassesSquared(snap, get_dependencies=False):
    if get_dependencies:
        return ['0_Temperatures', '0_Masses']
    return snap['0_Temperatures'] * snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
```
After setting this up, the user would be able to run e.g.

```
snap = pa.Snapshot(root_dir + '/data', 247)

# Automatically compute temperature times masses^2 using the user supplied function
snap['0_TM2']
```

## Openmp parallel execution of code

Paicos will upon startup check how many cores are available on your system.
This might be limited to the number set in the environment variable OMP_NUM_THREADS.

It is therefore sometimes useful to set the environment variable OMP_NUM_THREADS,
which will then be the maximum number of threads that Paicos can use.
```
export OMP_NUM_THREADS=16
```

You can also let Paicos know how many cores you would like to use, for instance
```
pa.numthreads(24)
```
would use 24 cores in the parts of the code that are parallelized.

## Setting up user settings

You can save a `.paicos_user_settings.py` script as a hidden file in your home directory,
which will then be imported when you do `import paicos`.
If you are unsure where to put it, then you can find the correct filepath by executing this block:

```
import paicos as pa
print('The recommended filepath for your .paicos_user_settings.py is:\n', pa.home_dir + '/.paicos_user_settings.py')
print('\nAlternatively, but no longer recommended due to high risk of accidental deletion, the filepath could be:\n', pa.code_dir + '/paicos_user_settings.py')
```

We include an example named `paicos_user_settings_template.py`, which you can use to get started.
It looks like this

```
import paicos as pa

"""
Set up your own default settings by renaming this file as paicos_user_settings.py
and saving it at the directory found at: pa.code_dir

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

```
