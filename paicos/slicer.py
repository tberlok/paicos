import numpy as np
from paicos import ImageCreator
from paicos import util


class Slicer(ImageCreator):
    """
    This implements slicing of gas variables.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, numthreads=16, make_snap_with_selection=False):
        from scipy.spatial import KDTree

        if make_snap_with_selection:
            raise ('make_snap_with_selection not yet implemented!')

        super().__init__(snap, center, widths, direction, npix=npix,
                         numthreads=numthreads)

        for ii, direc in enumerate(['x', 'y', 'z']):
            if self.direction == direc:
                assert self.widths[ii] == 0.

        # Pre-select a narrow region around the region-of-interest
        pos = np.array(snap["0_Coordinates"], dtype=np.float64)

        thickness = np.array(4.0*np.cbrt((snap["0_Volumes"]) /
                             (4.0*np.pi/3.0)), dtype=np.float64)

        self.slice = util.get_index_of_slice_region(pos, center, widths,
                                                    thickness, snap.box)

        # Now construct the image grid
        def unflatten(arr):
            return arr.flatten().reshape((npix_height, npix_width))

        extent = self.extent
        center = self.center

        npix_width = npix
        width = extent[1] - extent[0]
        height = extent[3] - extent[2]

        # TODO: Make assertion that dx=dy
        npix_height = int(height/width*npix_width)

        w = extent[0] + (np.arange(npix_width) + 0.5)*width/npix_width
        h = extent[2] + (np.arange(npix_height) + 0.5)*height/npix_height

        ww, hh = np.meshgrid(w, h)
        w = ww.flatten()
        h = hh.flatten()

        np.testing.assert_array_equal(ww, unflatten(ww.flatten()))

        ones = np.ones(w.shape[0])
        if direction == 'x':
            image_points = np.vstack([ones*center[0], w, h]).T
        elif direction == 'y':
            image_points = np.vstack([w, ones*center[1], h]).T
        elif direction == 'z':
            image_points = np.vstack([w, h, ones*center[2]]).T

        # Construct a tree and find the Voronoi cells closest to the image grid
        self.pos = pos[self.slice]
        tree = KDTree(self.pos)

        d, i = tree.query(image_points, workers=numthreads)

        self.index = unflatten(np.arange(pos.shape[0])[self.slice][i])
        self.distance_to_nearest_cell = unflatten(d)

    def get_image(self, variable):
        from warnings import warn
        warn('This method will be soon deprecated. in favor of the ' +
             ' method with name: slice_variable',
             DeprecationWarning, stacklevel=2)

        return self.slice_variable(variable)

    def slice_variable(self, variable):

        if isinstance(variable, str):
            variable = self.snap[variable]
        else:
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        return variable[self.index]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import paicos as pa
    from paicos import root_dir
    from matplotlib.colors import LogNorm

    pa.use_units(True)

    snap = pa.Snapshot(root_dir + '/data', 247)
    center = snap.Cat.Group['GroupPos'][0]

    width_vec = (
        [0.0, 10000, 10000],
        [10000, 0.0, 10000],
        [10000, 10000, 0.0],
        )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3)
    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        slicer = Slicer(snap, center, widths, direction, npix=512)

        image_filename = root_dir + '/data/slice_{}.hdf5'.format(direction)
        image_file = pa.ArepoImage(image_filename, slicer)

        Density = slicer.slice_variable(snap['0_Density'])
        Temperatures = slicer.slice_variable('0_Temperatures')

        image_file.save_image('Density', Density)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Make a plot
        if pa.units.enabled:
            axes[ii].imshow(Density.value, origin='lower',
                            extent=slicer.extent.value, norm=LogNorm())
        else:
            axes[ii].imshow(Density, origin='lower',
                            extent=slicer.extent, norm=LogNorm())

    plt.show()
