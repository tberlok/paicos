
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
