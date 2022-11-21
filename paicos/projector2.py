import numpy as np
from paicos import ImageCreator


class Projector2(ImageCreator):
    """
    This implements an SPH-like projection of gas variables.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, nvol=8, numthreads=16):
        from paicos import get_index_of_region_plus_thin_layer as get_index

        super().__init__(snap, center, widths, direction, npix=npix,
                         numthreads=numthreads)

        self.nvol = nvol

        self._check_if_omp_has_issues(numthreads)

        snap = self.snap
        xc = self.xc
        yc = self.yc
        zc = self.zc
        width_x = self.width_x
        width_y = self.width_y
        width_z = self.width_z

        snap.get_volumes()
        snap.load_data(0, "Coordinates")
        self.pos = pos = np.array(snap.P["0_Coordinates"], dtype=np.float64)

        self.hsml = np.array(np.cbrt(nvol*(snap.P["0_Volumes"]) /
                             (4.0*np.pi/3.0)), dtype=np.float64)

        if self.direction == 'x':
            self.index = get_index(pos, xc, yc, zc,
                                   width_x, width_y, width_z, self.hsml,
                                   snap.box)

        elif self.direction == 'y':
            self.index = get_index(pos, xc, yc, zc,
                                   width_x, width_y, width_z, self.hsml,
                                   snap.box)

        elif self.direction == 'z':
            self.index = get_index(pos, xc, yc, zc,
                                   width_x, width_y, width_z, self.hsml,
                                   snap.box)

        self.hsml = np.array(self.hsml[self.index])

        self.pos = self.pos[self.index]

    def _check_if_omp_has_issues(self, numthreads):
        from paicos import simple_reduction
        n = simple_reduction(1000, numthreads)
        if n == 1000:
            self.use_omp = True
            self.numthreads = numthreads
        else:
            self.use_omp = False
            self.numthreads = 1
            import warnings
            msg = ("OpenMP is seems to have issues with reduction operators" +
                   "on your system, so we'll turn it off." +
                   "If you're on Mac then the issue is likely a" +
                   "compiler problem, discussed here:\n" +
                   "https://stackoverflow.com/questions/54776301/" +
                   "cython-prange-is-repeating-not-parallelizing")
            warnings.warn(msg)

    def _get_variable(self, variable_str):

        from paicos import get_variable

        return get_variable(self.snap, variable_str)

    def project_variable(self, variable):

        if self.use_omp:
            from paicos import project_image2_omp as project_image2
        else:
            from paicos import project_image2

        import types
        if isinstance(variable, str):
            variable = self._get_variable(variable)
        elif isinstance(variable, types.FunctionType):
            variable = variable(self.arepo_snap)
        elif isinstance(variable, np.ndarray):
            pass
        else:
            raise RuntimeError('Unexpected type for variable')

        variable = np.array(variable[self.index], dtype=np.float64)

        xc = self.xc
        yc = self.yc
        zc = self.zc
        boxsize = self.snap.box
        if self.direction == 'x':
            projection = project_image2(self.pos[:, 1],
                                       self.pos[:, 2],
                                       self.pos[:, 0],
                                       variable,
                                       self.hsml, self.npix,
                                       yc, zc, xc, self.width_y, self.width_z, self.width_x,
                                       boxsize, self.numthreads)
        elif self.direction == 'y':
            projection = project_image2(self.pos[:, 0],
                                       self.pos[:, 2],
                                       self.pos[:, 1],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, zc, yc, self.width_x, self.width_z, self.width_y,
                                       boxsize, self.numthreads)
        elif self.direction == 'z':
            projection = project_image2(self.pos[:, 0],
                                       self.pos[:, 1],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, yc, zc, self.width_x, self.width_y, self.width_z,
                                       boxsize, self.numthreads)
        return projection.T


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from paicos import Snapshot
    from paicos import ArepoImage
    from paicos import root_dir

    snap = Snapshot(root_dir + '/data', 247)
    center = snap.Cat.Group['GroupPos'][0]
    R200c = snap.Cat.Group['Group_R_Crit200'][0]

    width_vec = (
        [0.25*R200c, 10000, 20000],
        [10000, 0.25*R200c, 20000],
        [10000, 20000, 0.25*R200c],
        )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3)
    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        projector = Projector2(snap, center, widths, direction, npix=512)

        filename = root_dir + '/data/projection_{}.hdf5'.format(direction)
        image_file = ArepoImage(filename, projector)

        Masses = projector.project_variable('Masses')
        Volume = projector.project_variable('Volumes')

        image_file.save_image('Masses', Masses)
        image_file.save_image('Volumes', Volume)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Make a plot
        axes[ii].imshow(np.log10(Masses/Volume), origin='lower',
                        extent=projector.extent)
    plt.show()
