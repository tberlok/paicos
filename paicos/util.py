
def get_project_root_dir():
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''
    path_split = path.split('/')
    for ii in range(1, len(path_split)-1):
        root_dir += '/' + path_split[ii]

    return root_dir


root_dir = get_project_root_dir()


def save_dataset(hdf5file, name, data=None):
    """
    Create dataset in *open* hdf5file ( hdf5file = h5py.File(filename, 'r') )
    If the data has units then they are saved as an attribute.
    """
    if hasattr(data, 'unit'):
        hdf5file.create_dataset(name, data=data.value)
        attrs = {'unit': data.unit.to_string()}
        for key in attrs.keys():
            hdf5file[name].attrs[key] = attrs[key]
    else:
        hdf5file.create_dataset(name, data=data)


def load_dataset(hdf5file, name):
    """
    Load dataset, returning a paicos quantity if the attributes
    contain units
    """
    raise RuntimeError('Needs to be implemented!')
    pass


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
        new_kwargs = dict(kwargs)
        for key in kwargs.keys():
            if hasattr(kwargs[key], 'unit'):
                new_kwargs[key] = kwargs[key].value

        return func(*new_args, **new_kwargs)
    return inner
