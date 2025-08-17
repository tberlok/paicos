"""
Defines a few useful functions and most importantly, the low level
hdf5 reader and writer. It also serves as a placeholder for variables
that can be changed via user functions.
"""
import os
import warnings
import numpy as np
import h5py
from functools import wraps
import time
from . import settings
from . import units as pu
from .cython.get_index_of_region import get_cube, get_radial_range
from .cython.get_index_of_region import get_cube_plus_thin_layer
from .cython.get_index_of_region import get_radial_range_plus_thin_layer
from .cython.get_index_of_region import get_rotated_cube
from .cython.get_index_of_region import get_rotated_cube_plus_thin_layer
from .cython.openmp_info import simple_reduction, get_openmp_settings

# These will be set by the user using the add_user_unit function
user_unit_dict = {'default': {},
                  'voronoi_cells': {},
                  'dark_matter': {},
                  'stars': {},
                  'black_holes': {},
                  'groups': {},
                  'subhalos': {}}


def get_project_root_dir():
    """
    Returns the root directory of the local Paicos copy.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    r_dir = ''
    path_split = path.split('/')
    for ii in range(1, len(path_split) - 1):
        r_dir += '/' + path_split[ii]

    return r_dir + '/'


# Set the root_dir for Paicos
root_dir = get_project_root_dir()
home_dir = os.path.expanduser('~') + '/'


def _split_filename(filenamewithpath):
    """
    A convenience function for getting the basedir, basename and snapsnum
    from a filename including its path.

    Example usage:
        print(split_filename('data/small_non_comoving_007.hdf5'))
        print(split_filename('small_non_comoving.hdf5'))
        print(split_filename('small_non_comoving.0.hdf5'))

    """
    err_msg = "needs to be a hdf5 file, .e.g. filenamewithpath = 'data/snap_010.hdf5'"

    assert filenamewithpath[-5:] == '.hdf5', err_msg

    snapnum = None

    index = filenamewithpath[::-1].find('_')
    if index != -1:
        underscore_loc = -index - 1

    if filenamewithpath[underscore_loc] == '_':
        snapnum_str = filenamewithpath[underscore_loc + 1:underscore_loc + 4]
        if np.char.isdigit(snapnum_str):
            snapnum = int(snapnum_str)

    filename = os.path.basename(filenamewithpath)
    basedir = filenamewithpath.replace(filename, '')

    if snapnum is None:
        basename = filename[:-5]
    else:
        basename = filename[:underscore_loc]

    return basedir, basename, snapnum


def save_dataset(hdf5file, name, data=None, data_attrs={},
                 group=None, group_attrs={}):
    """
    Create dataset in *open* hdf5file ( hdf5file = h5py.File(filename, 'w') )
    If the data has units then they are saved as an attribute.
    """
    assert isinstance(data_attrs, dict)
    data_attrs = dict(data_attrs)
    assert isinstance(group_attrs, dict)
    group_attrs = dict(group_attrs)

    # Allow for storing data sets in groups or nested groups
    if group is None:
        path = hdf5file
    else:
        if group not in hdf5file:
            hdf5file.create_group(group)
            for key in group_attrs:
                hdf5file[group].attrs[key] = group_attrs[key]
        path = hdf5file[group]

    # Save data set
    if hasattr(data, 'unit'):
        path.create_dataset(name, data=data.value)
    else:
        path.create_dataset(name, data=data)

    # Write attributes
    if isinstance(data, pu.PaicosTimeSeries):
        data_attrs.update(data.hdf5_attrs)
    elif hasattr(data, 'unit'):
        data_attrs['unit'] = data.unit.to_string()
    for key in data_attrs.keys():
        path[name].attrs[key] = data_attrs[key]

    # Check if scale_factors are already saved
    if isinstance(data, pu.PaicosTimeSeries):
        if 'scale_factor' not in path.keys():
            path.create_dataset('scale_factor', data=data.a)
        else:
            err_msg = ("All PaicosTimeSeries need to have the same array "
                       + "of scale factors when saving to the same file")
            np.testing.assert_allclose(path['scale_factor'][...], data.a,
                                       err_msg=err_msg, rtol=1e-14, atol=1e-14)


def load_dataset(hdf5file, name, group=None):
    """
    Load dataset, returning a paicos quantity if the attributes
    contain units and units are enabled.
    """

    if not isinstance(hdf5file, h5py.File):
        if isinstance(hdf5file, str):
            hdf5file = h5py.File(hdf5file, 'r')
        else:
            msg = "'util.load_dataset' needs a file name or an open hdf5 file"
            raise RuntimeError(msg)

    comoving_sim = bool(hdf5file['Parameters'].attrs['ComovingIntegrationOn'])
    time = hdf5file['Header'].attrs['Time']
    hubble_param = hdf5file['Parameters'].attrs['HubbleParam']
    if hubble_param == 0.:
        hubble_param = 1.0

    # Allow for loading data sets in groups or nested groups
    if group is None:
        path = hdf5file
    else:
        if group not in hdf5file:
            hdf5file.create_group(group)
        path = hdf5file[group]

    data = path[name][()]

    if settings.use_units:
        if 'unit' in path[name].attrs.keys():
            unit = path[name].attrs['unit']
            if 'Paicos' in path[name].attrs.keys():
                if path[name].attrs['Paicos'] == 'PaicosTimeSeries':
                    if comoving_sim:
                        time = path['scale_factor'][...]
                    else:
                        time = path['time'][...]
                    data = pu.PaicosTimeSeries(data, unit, a=time, h=hubble_param,
                                               comoving_sim=comoving_sim)
                elif path[name].attrs['Paicos'] == 'PaicosQuantity':
                    data = pu.PaicosQuantity(data, unit, a=time, h=hubble_param,
                                             comoving_sim=comoving_sim)
            else:
                data = pu.PaicosQuantity(data, unit, a=time, h=hubble_param,
                                         comoving_sim=comoving_sim, copy=True)
    return data


def remove_astro_units(func):
    """
    This is a decorator function that takes in a function and returns a new
    function that removes any astro units from the function's arguments and
    keyword arguments before calling the original function. The decorator
    checks if the argument or keyword argument has a 'unit' attribute and, if
    so, replaces it with its 'value' attribute. This is useful for functions
    that do not support astro units or need to work with raw values.
    """
    @wraps(func)
    def remove_astro_units_inner(*args, **kwargs):

        unit_dict = {'length': []}
        # Create new args
        new_args = list(args)
        for ii, new_arg in enumerate(new_args):
            if hasattr(new_arg, 'unit'):
                new_args[ii] = new_arg.value
                if isinstance(new_arg, pu.PaicosQuantity):
                    phys_type = new_arg.uq.to_physical.unit.physical_type
                    unit = new_arg.uq.unit
                else:
                    phys_type = new_arg.unit.physical_type
                    unit = new_arg.unit
                if str(phys_type) == 'length':
                    unit_dict['length'].append(unit)

        for ii in range(len(unit_dict['length'])):
            if unit_dict['length'][ii] != unit_dict['length'][0]:
                err_msg = ("Paicos: Not all quantities with physical type "
                           + "'length' have the same units. Stopping here "
                           + "to prevent unit-related errors.")
                print("Contents of unit_dict['length']", unit_dict['length'])
                raise RuntimeError(err_msg)

        # Create new kwargs
        new_kwargs = kwargs  # dict(kwargs)
        for key, kwarg in kwargs.items():
            if hasattr(kwarg, 'unit'):
                new_kwargs[key] = kwarg.value

        return func(*new_args, **new_kwargs)
    return remove_astro_units_inner


@remove_astro_units
def get_index_of_radial_range(pos, center, r_min, r_max):
    """
    Get a boolean array of positions, pos, which are inside the spherical
    shell with inner radius r_min and outer radius r_max, centered at center.
    """
    assert center.shape[0] == 3
    x_c, y_c, z_c = center[0], center[1], center[2]
    index = get_radial_range(pos, x_c, y_c, z_c, r_min, r_max,
                             settings.numthreads)
    return index


@remove_astro_units
def get_index_of_radial_range_plus_thin_layer(pos, center, r_min, r_max, thickness):
    """
    Get a boolean array for the positions, pos, which are inside the spherical
    shell with inner radius r_min and outer radius r_max, centered at center.
    The thickness array argument adds a variable thickness, e.g., the cell
    diameters, so that point close to the selection are also included.
    """
    assert center.shape[0] == 3
    x_c, y_c, z_c = center[0], center[1], center[2]
    index = get_radial_range_plus_thin_layer(pos, x_c, y_c, z_c, r_min, r_max,
                                             thickness, settings.numthreads)
    return index


@remove_astro_units
def get_index_of_cubic_region(pos, center, widths, box):
    """
    Get a boolean array to the position array, pos, which are inside a cubic
    region.

    Parameters
    ----------
    pos : array
         position array with dimensions = (n, 3)
    center : array with length 3
             The center of the box (x, y, z).
    widths : array with length 3
             The widths of the box.
    box : float
                 The box size of the simulation (e.g. snap.box).
    """
    x_c, y_c, z_c = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    index = get_cube(pos, x_c, y_c, z_c, width_x, width_y, width_z, box,
                     settings.numthreads)
    return index


@remove_astro_units
def get_index_of_cubic_region_plus_thin_layer(pos, center, widths, thickness, box):
    """
    Get a boolean array to the position array, pos, which are inside a cubic
    region plus a thin layer with a cell-dependent thickness

    Parameters
    ----------
    pos : array
         position array with dimensions = (n, 3)
    center : array with length 3
             The center of the box (x, y, z).
    widths : array with length 3
             The widths of the box.
    thickness : array
                Array with same length as the position array.
    box : float
         The box size of the simulation (e.g. snap.box).
    """
    x_c, y_c, z_c = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    index = get_cube_plus_thin_layer(pos, x_c, y_c, z_c, width_x, width_y,
                                     width_z, thickness, box, settings.numthreads)
    return index


@remove_astro_units
def get_index_of_rotated_cubic_region(pos, center, widths, box, orientation):
    """
    Get a boolean array to the position array, pos, which are inside a cubic
    region

    Parameters
    ----------
    pos : array
         position array with dimensions = (n, 3)
    center : array with length 3
             The center of the box (x, y, z).
    widths : array with length 3
             The widths of the box.
    thickness : array
                Array with same length as the position array.
    box : float
         The box size of the simulation (e.g. snap.box).
    """
    x_c, y_c, z_c = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    unit_vectors = orientation.cartesian_unit_vectors
    index = get_rotated_cube(pos, x_c, y_c, z_c, width_x, width_y,
                             width_z, box,
                             unit_vectors['x'],
                             unit_vectors['y'],
                             unit_vectors['z'],
                             settings.numthreads)
    return index


@remove_astro_units
def get_index_of_rotated_cubic_region_plus_thin_layer(pos, center, widths, thickness,
                                                      box, orientation):
    """
    Get a boolean array to the position array, pos, which are inside a cubic
    region plus a thin layer with a cell-dependent thickness

    Parameters
    ----------
    pos : array
         position array with dimensions = (n, 3)
    center : array with length 3
             The center of the box (x, y, z).
    widths : array with length 3
             The widths of the box.
    thickness : array
                Array with same length as the position array.
    box : float
         The box size of the simulation (e.g. snap.box).
    """
    x_c, y_c, z_c = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    unit_vectors = orientation.cartesian_unit_vectors
    index = get_rotated_cube_plus_thin_layer(pos, x_c, y_c, z_c, width_x, width_y,
                                             width_z, thickness, box,
                                             unit_vectors['x'],
                                             unit_vectors['y'],
                                             unit_vectors['z'],
                                             settings.numthreads)
    return index


def _check_if_omp_has_issues(verbose=True):
    """
    Check if the parallelization via OpenMP works.
    Parameters
    ----------
    numthreads : int
        Number of threads used in parallelization
    """

    if settings.give_openMP_warnings is False:
        verbose = False

    # max_threads = get_openmp_settings(0, True)
    max_threads = os.cpu_count()
    settings.max_threads = max_threads

    if settings.numthreads > max_threads and verbose:
        msg = ('\n\nThe default number of OpenMP threads, {}, '
               + 'exceeds the {} available on your system. Setting '
               + 'numthreads={}. '
               + 'You can set numthreads with e.g. the command\n '
               + 'paicos.set_numthreads(16)\n\n')
        print(msg.format(settings.numthreads, max_threads, max_threads))
        settings.numthreads = max_threads

    n = simple_reduction(1000, settings.numthreads)
    if n == 1000:
        settings.numthreads_reduction = settings.numthreads
        settings.openMP_has_issues = False
    else:
        # We have issues...
        settings.openMP_has_issues = True
        settings.numthreads_reduction = 1

        msg = ("OpenMP seems to have issues with reduction operators "
               + "on your system, so we'll turn it off for those use cases. "
               + "If you're on Mac then the issue is likely a "
               + "compiler problem, discussed here:\n"
               + "https://stackoverflow.com/questions/54776301/"
               + "cython-prange-is-repeating-not-parallelizing.\n\n")
        if verbose:
            warnings.warn(msg)


def _copy_over_snapshot_information(snap, new_filename, mode='r+'):
    """
    Copy over attributes from the original arepo snapshot.
    In this way we will have access to units used, redshift etc
    """
    info_dic = {}
    info_dic['Header'] = dict(snap.Header)
    info_dic['Parameters'] = dict(snap.Parameters)
    info_dic['Config'] = dict(snap.Config)
    with h5py.File(new_filename, mode) as f:
        for group in ['Header', 'Parameters', 'Config']:
            f.create_group(group)
            for key in info_dic[group]:
                f[group].attrs[key] = info_dic[group][key]


def conditional_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timing = kwargs.pop('timing', False)
        if timing:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            class_info = ""
            if args and hasattr(args[0], '__class__'):
                class_info = f"{args[0].__class__.__name__}."

            print(f"[TIMER] {class_info}{func.__name__} took {end - start:.6f} seconds")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper
