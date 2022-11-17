import numpy as np


class Projector:
    """
    This implements an SPH-like projection of gas variables.
    """

    def __init__(self, arepo_snap, center, widths, direction,
                 npix=512, nvol=8, numthreads=16):
        from paicos import get_index_of_region
        from paicos import simple_reduction

        n = simple_reduction(1000, numthreads)
        if n == 1000:
            self.use_omp = True
            self.numthreads = numthreads
        else:
            self.use_omp = False
            self.numthreads = 1
            print('OpenMP is not working on your system...')

        self.snap = arepo_snap

        self.center = center
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]

        self.widths = widths
        self.width_x = widths[0]
        self.width_y = widths[1]
        self.width_z = widths[2]

        self.direction = direction

        self.npix = npix

        snap = arepo_snap
        xc = self.xc
        yc = self.yc
        zc = self.zc
        width_x = self.width_x
        width_y = self.width_y
        width_z = self.width_z

        snap.get_volumes()
        snap.load_data(0, "Coordinates")
        self.pos = pos = np.array(snap.P["0_Coordinates"], dtype=np.float64)

        if direction == 'x':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.yc - self.width_y/2, self.yc + self.width_y/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'y':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'z':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.yc - self.width_y/2, self.yc + self.width_y/2]

        self.hsml = np.cbrt(nvol*(snap.P["0_Volumes"][self.index]) /
                            (4.0*np.pi/3.0))

        self.hsml = np.array(self.hsml, dtype=np.float64)

        self.pos = self.pos[self.index]

    def _get_variable(self, variable_str):

        from paicos import get_variable

        return get_variable(self.snap, variable_str)

    def project_variable(self, variable):

        if self.use_omp:
            from paicos import project_image_omp as project_image
        else:
            from paicos import project_image

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
            projection = project_image(self.pos[:, 1],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       yc, zc, self.width_y, self.width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'y':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, zc, self.width_x, self.width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'z':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 1],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, yc, self.width_x, self.width_y,
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
    # widths = [10000, 10000, 2*R200c]
    widths = [10000, 10000, 10000]
    width_vec = (
        [2*R200c, 10000, 20000],
        [10000, 2*R200c, 20000],
        [10000, 20000, 2*R200c],
        )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3)
    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        p = Projector(snap, center, widths, direction, npix=512)

        filename = root_dir + '/data/projection_{}.hdf5'.format(direction)
        image_file = ArepoImage(filename, snap.first_snapfile_name, center,
                                widths, direction)

        Masses = p.project_variable('Masses')
        Volume = p.project_variable('Volumes')

        image_file.save_image('Masses', Masses)
        image_file.save_image('Volumes', Volume)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Make a plot
        axes[ii].imshow(np.log10(Masses/Volume), origin='lower',
                        extent=p.extent)
    plt.show()
