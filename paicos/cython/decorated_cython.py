
def remove_astro_units(func):
    def inner(*args, **kwargs):
        tmp_args = list(args)
        for ii in range(len(tmp_args)):
            if hasattr(tmp_args[ii], 'unit'):
                tmp_args[ii] = tmp_args[ii].value
                # tmp_args[ii][100:200] *= 4
        args = tuple(tmp_args)

        # print('Arguments for args: {}'.format(args))
        # print('Arguments for kwargs: {}'.format(kwargs))
        out = func(*args, **kwargs)
        return out
    return inner


def get_index_of_region(pos, xc, yc, zc, sidelength_x, sidelength_y,
                        thickness, boxsize):
    from .cython import get_index_of_region

    # Do all the  the boring checks here before passing,
    # Or perhaps write a decorator which takes care of things?
    # Could try out both and make a check for speed!
    # The indexing functions need a decorator which automatically
    # removes units

    return index


# def project_image(ddffd , use_omp):



