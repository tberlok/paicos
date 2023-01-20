import numpy as np
from paicos import ImageCreator


class TreeProjector(ImageCreator):
    """
    This implements projection of gas variables using approximate
    integration. Probably mostly useful for thin projections.
    """

    def __init__(self, snap, center, widths, direction, npix_depth,
                 npix=512, numthreads=16):

        from scipy.spatial import KDTree

        super().__init__(snap, center, widths, direction, npix=npix,
                         numthreads=numthreads)

        self.npix_depth = npix_depth

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

        thickness = np.array(4.0*np.cbrt((snap.P["0_Volumes"]) /
                             (4.0*np.pi/3.0)), dtype=np.float64)

        if direction == 'x':
            from paicos import get_index_of_x_slice_region
            thickness += width_x
            depth = width_x
            start = self.xc - self.width_x/2
            self.slice = get_index_of_x_slice_region(pos, xc, yc, zc,
                                                     width_y, width_z,
                                                     thickness, snap.box)
        elif direction == 'y':
            from paicos import get_index_of_y_slice_region
            thickness += width_y
            depth = width_y
            start = self.yc - self.width_y/2
            self.slice = get_index_of_y_slice_region(pos, xc, yc, zc,
                                                     width_x, width_z,
                                                     thickness, snap.box)

        elif direction == 'z':
            from paicos import get_index_of_z_slice_region
            thickness += width_z
            depth = width_z
            start = self.zc - self.width_z/2
            self.slice = get_index_of_z_slice_region(pos, xc, yc, zc,
                                                     width_x, width_y,
                                                     thickness, snap.box)

        # Now construct the image grid

        def unflatten(arr):
            # if direction == 'x':
            #     shape = (npix_depth, npix_height, npix_width)
            # elif direction == 'y':
            shape = (npix_width, npix_height, npix_depth)
            # elif direction == 'z':
            # shape = (npix_depth, npix_height, npix_width)
            return arr.flatten().reshape(shape)

        npix_width = npix
        width = self.extent[1] - self.extent[0]
        height = self.extent[3] - self.extent[2]

        # TODO: Make assertion that dx=dy
        npix_height = int(height/width*npix_width)

        w = self.extent[0] + (np.arange(npix_width) + 0.5)*width/npix_width
        h = self.extent[2] + (np.arange(npix_height) + 0.5)*height/npix_height
        d = start + (np.arange(npix_depth) + 0.5)*depth/npix_depth

        ww, hh, dd = np.meshgrid(w, h, d)
        w = ww.flatten()
        h = hh.flatten()
        d = hh.flatten()

        # np.testing.assert_array_equal(ww, unflatten(ww.flatten()))

        if direction == 'x':
            image_points = np.vstack([d, w, h]).T
        elif direction == 'y':
            image_points = np.vstack([w, d, h]).T
        elif direction == 'z':
            image_points = np.vstack([w, h, d]).T

        # Construct a tree and find the Voronoi cells closest to the image grid
        self.pos = pos[self.slice]
        tree = KDTree(self.pos)

        d, i = tree.query(image_points, workers=numthreads)

        self.index = unflatten(np.arange(pos.shape[0])[self.slice][i])
        # self.distance_to_nearest_cell = unflatten(d)

    def _get_variable(self, variable_str):

        from paicos import get_variable

        return get_variable(self.snap, variable_str)

    def project_variable(self, variable):

        import types
        if isinstance(variable, str):
            variable = self._get_variable(variable)
        elif isinstance(variable, types.FunctionType):
            variable = variable(self.arepo_snap)
        elif isinstance(variable, np.ndarray):
            pass
        else:
            raise RuntimeError('Unexpected type for variable')

        # variable = np.array(variable[self.index], dtype=np.float64)
        # if direction == 'x':
        #     image = np.sum(variable[self.index], axis=0)/self.npix_depth
        # elif direction == 'y':
        #     image = np.sum(variable[self.index], axis=1)/self.npix_depth
        # elif direction == 'z':
        image = np.sum(variable[self.index], axis=2)/self.npix_depth
        return image.T


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
        projector = TreeProjector(snap, center, widths, direction,
                                  npix_depth=8, npix=128)

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
