
use_paicos_quantities = True


def get_project_root_dir():
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''
    path_split = path.split('/')
    for ii in range(1, len(path_split)-1):
        root_dir += '/' + path_split[ii]

    return root_dir


root_dir = get_project_root_dir()
