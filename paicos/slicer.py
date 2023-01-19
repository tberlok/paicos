import numpy as np
from paicos import ImageCreator


class Slicer(ImageCreator):
    """
    This implements slicing of gas variables.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, numthreads=16):

        from scipy.spatial import KDTree
        from paicos import units

        super().__init__(snap, center, widths, direction, npix=npix,
                         numthreads=numthreads)

        snap = self.snap
        if units.enabled:
            xc = self.xc.value
            yc = self.yc.value
            zc = self.zc.value
            width_x = self.width_x.value
            width_y = self.width_y.value
            width_z = self.width_z.value
            extent = self.extent.value
            center = self.center.value
        else:
            xc = self.xc
            yc = self.yc
            zc = self.zc
            width_x = self.width_x
            width_y = self.width_y
            width_z = self.width_z
            extent = self.extent
            center = self.center

        for ii, direc in enumerate(['x', 'y', 'z']):
            if self.direction == direc:
                assert self.widths[ii] == 0.

        # Pre-select a narrow region around the region-of-interest
        # snap.get_volumes()
        # snap.load_data(0, "Coordinates")
        pos = np.array(snap["0_Coordinates"], dtype=np.float64)

        thickness = np.array(4.0*np.cbrt((snap["0_Volumes"]) /
                             (4.0*np.pi/3.0)), dtype=np.float64)

        if direction == 'x':
            from paicos import get_index_of_x_slice_region
            self.slice = get_index_of_x_slice_region(pos, xc, yc, zc,
                                                     width_y, width_z,
                                                     thickness, snap.box)

        elif direction == 'y':
            from paicos import get_index_of_y_slice_region
            self.slice = get_index_of_y_slice_region(pos, xc, yc, zc,
                                                     width_x, width_z,
                                                     thickness, snap.box)

        elif direction == 'z':
            from paicos import get_index_of_z_slice_region
            self.slice = get_index_of_z_slice_region(pos, xc, yc, zc,
                                                     width_x, width_y,
                                                     thickness, snap.box)

        # Now construct the image grid

        def unflatten(arr):
            return arr.flatten().reshape((npix_height, npix_width))

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

        if direction == 'x':
            image_points = np.vstack([np.ones_like(w)*center[0], w, h]).T
        elif direction == 'y':
            image_points = np.vstack([w, np.ones_like(w)*center[1], h]).T
        elif direction == 'z':
            image_points = np.vstack([w, h, np.ones_like(w)*center[2]]).T

        # Construct a tree and find the Voronoi cells closest to the image grid
        self.pos = pos[self.slice]
        tree = KDTree(self.pos)

        d, i = tree.query(image_points, workers=numthreads)

        self.index = unflatten(np.arange(pos.shape[0])[self.slice][i])
        self.distance_to_nearest_cell = unflatten(d)

    def get_image(self, variable):
        from warnings import warn
        warn('This method will be soon deprecated. in favour of the ' +
             ' method with name: slice_variable',
             DeprecationWarning, stacklevel=2)

        return variable[self.index]

    def slice_variable(self, variable):

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

        # snap.load_data(0, 'Density')
        # snap.load_data(0, 'Velocities')
        # snap.load_data(0, 'MagneticField')

        Density = slicer.slice_variable(snap['0_Density'])

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

    if not pa.units.enabled:
        # Also an example where we slice a Paicos Quantity with units
        rho = snap.converter.get_paicos_quantity(snap['0_Density'],
                                                 'Density')
        density_slice = slicer.slice_variable(rho)
        from astropy import constants as c
        plt.figure(2)
        density_slice = (density_slice.cgs/c.m_p.cgs).to_physical
        plt.imshow(density_slice.value, norm=LogNorm())
        plt.title(density_slice.label('n'))

    plt.show()
