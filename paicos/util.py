from . import settings

user_functions = {}

openMP_has_issues = None


def get_project_root_dir():
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''
    path_split = path.split('/')
    for ii in range(1, len(path_split)-1):
        root_dir += '/' + path_split[ii]

    return root_dir + '/'


root_dir = get_project_root_dir()


def save_dataset(hdf5file, name, data=None, group=None, group_attrs=None):
    """
    Create dataset in *open* hdf5file ( hdf5file = h5py.File(filename, 'w') )
    If the data has units then they are saved as an attribute.
    """

    # Allow for storing data sets in groups or nested groups
    if group is None:
        path = hdf5file
    else:
        if group not in hdf5file:
            hdf5file.create_group(group)
            if isinstance(group_attrs, dict):
                for key in group_attrs.keys():
                    hdf5file[group].attrs[key] = group_attrs[key]
        path = hdf5file[group]

    # Save data set
    if hasattr(data, 'unit'):
        path.create_dataset(name, data=data.value)
        attrs = {'unit': data.unit.to_string()}
        for key in attrs.keys():
            path[name].attrs[key] = attrs[key]
    else:
        path.create_dataset(name, data=data)


def load_dataset(hdf5file, name, converter=None, group=None):
    """
    Load dataset, returning a paicos quantity if the attributes
    contain units and units are enabled.
    """
    import h5py

    if not isinstance(hdf5file, h5py._hl.files.File):
        if isinstance(hdf5file, str):
            hdf5file = h5py.File(hdf5file, 'r')
        else:
            msg = "'util.load_dataset' needs a file name or an open hdf5 file"
            raise RuntimeError(msg)

    # Construct a convertor object if it was not passed
    if converter is None:
        from .arepo_converter import ArepoConverter
        converter = ArepoConverter(hdf5file.filename)

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
            from . import units as pu
            unit = path[name].attrs['unit']
            data = pu.PaicosQuantity(data, unit, a=converter.a, h=converter.h)
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
    def inner(*args, **kwargs):
        # Create new args
        new_args = list(args)
        for ii in range(len(new_args)):
            if hasattr(new_args[ii], 'unit'):
                new_args[ii] = new_args[ii].value

        # Create new kwargs
        new_kwargs = kwargs  # dict(kwargs)
        for key in kwargs.keys():
            if hasattr(kwargs[key], 'unit'):
                new_kwargs[key] = kwargs[key].value

        return func(*new_args, **new_kwargs)
    return inner


@remove_astro_units
def get_index_of_radial_range(pos, center, r_min, r_max):
    from .cython.get_index_of_region_functions import get_index_of_radial_range as get_index_of_radial_range_cython
    xc, yc, zc = center[0], center[1], center[2]
    index = get_index_of_radial_range_cython(pos, xc, yc, zc, r_min, r_max)
    return index


@remove_astro_units
def get_index_of_region(pos, center, widths, box):
    from .cython.get_index_of_region_functions import get_index_of_region as get_index_of_region_cython
    xc, yc, zc = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    index = get_index_of_region_cython(pos, xc, yc, zc,
                                       width_x, width_y, width_z, box)
    return index


@remove_astro_units
def get_index_of_slice_region(pos, center, widths, thickness, box):
    xc, yc, zc = center[0], center[1], center[2]
    width_x, width_y, width_z = widths
    if widths[0] == 0.:
        from .cython.get_index_of_region_functions import get_index_of_x_slice_region
        index = get_index_of_x_slice_region(pos, xc, yc, zc,
                                            width_y, width_z,
                                            thickness, box)
    elif widths[1] == 0.:
        from .cython.get_index_of_region_functions import get_index_of_y_slice_region
        index = get_index_of_y_slice_region(pos, xc, yc, zc,
                                            width_x, width_z,
                                            thickness, box)
    elif widths[2] == 0.:
        from .cython.get_index_of_region_functions import get_index_of_z_slice_region
        index = get_index_of_z_slice_region(pos, xc, yc, zc,
                                            width_x, width_y,
                                            thickness, box)
    else:
        raise RuntimeError('width={} should have length 3 and contain a zero!')

    return index


def check_if_omp_has_issues(verbose=True):
    """
    Check if the parallelization via OpenMP works.

    Parameters
    ----------
    numthreads : int
        Number of threads used in parallelization
    """
    from .cython.openmp_info import simple_reduction, get_openmp_settings

    if openMP_has_issues is not None:
        return openMP_has_issues

    if settings.give_openMP_warnings is False:
        verbose = False

    max_threads = get_openmp_settings(0, False)
    if settings.numthreads > max_threads:
        msg = ('\n\nThe user specified number of OpenMP threads, {}, ' +
               'exceeds the {} available on your system. Setting ' +
               'numthreads to use half the available threads, i.e. {}.\n' +
               'You can set numthreads with e.g. the command\n ' +
               'paicos.set_numthreads(16)\n\n')
        print(msg.format(settings.numthreads, max_threads, max_threads//2))
        settings.numthreads = max_threads//2

    n = simple_reduction(1000, settings.numthreads)
    if n == 1000:
        return False
    else:
        import warnings
        msg = ("OpenMP seems to have issues with reduction operators " +
               "on your system, so we'll turn it off for those use cases. " +
               "If you're on Mac then the issue is likely a " +
               "compiler problem, discussed here:\n" +
               "https://stackoverflow.com/questions/54776301/" +
               "cython-prange-is-repeating-not-parallelizing.\n\n")
        if verbose:
            warnings.warn(msg)
        return True
