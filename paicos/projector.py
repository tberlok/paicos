import numpy as np
from paicos import ImageCreator
from paicos import util


class Projector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane.

    The Projector class is a subclass of the ImageCreator class.
    The Projector class creates an image of a given variable by projecting
    it onto a 2D plane.

    It takes in several parameters such as a snapshot object, center and
    widths of the region, direction of projection, and various optional
    parameters for number of pixels, smoothing length and number of threads
    for parallelization. It then calls various functions from the paicos
    package, including get_index_of_region, load_data, and get_volumes. This
    class also has a function called _check_if_omp_has_issues which checks if
    the parallelization via OpenMP works, and sets the number of threads
    accordingly.
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, nvol=8, numthreads=16):

        """
        Initialize the Projector class.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center : numpy array
            Center of the region on which projection is to be done, e.g.
            center = [xc, yc, zc].

        widths : numpy array
            Widths of the region on which projection is to be done,
            e.g.m widths=[width_x, width_y, width_z].

        direction : str
            Direction of the projection, e.g. 'x', 'y' or 'z'.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        nvol : int, optional
            Integer used to determine the smoothing length, by default 8

        numthreads : int, optional
            Number of threads used in parallelization, by default 16
        """

        # call the superclass constructor to initialize the ImageCreator class
        super().__init__(snap, center, widths, direction, npix=npix,
                         numthreads=numthreads)

        # nvol is an integer that determines the smoothing length
        self.nvol = nvol

        # check if OpenMP has any issues with the number of threads
        self._check_if_omp_has_issues(numthreads)

        snap = self.snap

        # get the cell positions from the snapshot
        self.pos = pos = np.array(snap["0_Coordinates"], dtype=np.float64)

        # get the index of the region of projection
        self.index = util.get_index_of_region(pos, center, widths, snap.box)

        # Calculate the smoothing length
        self.hsml = np.cbrt(nvol*(snap["0_Volumes"][self.index]) /
                            (4.0*np.pi/3.0))

        self.hsml = np.array(self.hsml, dtype=np.float64)

        self.pos = self.pos[self.index]

    def _check_if_omp_has_issues(self, numthreads):
        """
        Check if the parallelization via OpenMP works.

        Parameters
        ----------
        numthreads : int
            Number of threads used in parallelization
        """

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

    @util.remove_astro_units
    def _cython_project(self, center, widths, variable):
        if self.use_omp:
            from paicos import project_image_omp as project_image
        else:
            from paicos import project_image

        xc, yc, zc = center[0], center[1], center[2]
        width_x, width_y, width_z = widths

        boxsize = self.snap.box
        if self.direction == 'x':
            projection = project_image(self.pos[:, 1],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       yc, zc, width_y, width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'y':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 2],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, zc, width_x, width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'z':
            projection = project_image(self.pos[:, 0],
                                       self.pos[:, 1],
                                       variable,
                                       self.hsml, self.npix,
                                       xc, yc, width_x, width_y,
                                       boxsize, self.numthreads)

        return projection

    def project_variable(self, variable):
        """
        projects a given variable onto a 2D plane.

        Parameters
        ----------
        variable : str, function, numpy array
            variable, it can be passed as string or an array

        Returns
        -------
        numpy array
            The image of the projected variable
        """

        from paicos import units

        if isinstance(variable, str):
            variable = self.snap[variable]
        elif isinstance(variable, np.ndarray):
            pass
        else:
            raise RuntimeError('Unexpected type for variable')

        if isinstance(variable, units.PaicosQuantity):
            variable_unit = variable.unit
            variable = units.PaicosQuantity(variable[self.index],
                                            variable.unit,
                                            dtype=np.float64,
                                            a=self.snap.a, h=self.snap.h)
        else:
            variable = np.array(variable[self.index], dtype=np.float64)

        # Do the projection
        projection = self._cython_project(self.center, self.widths, variable)

        # Transpose
        projection = projection.T
        area_per_pixel = self.area/np.product(projection.shape)

        if isinstance(variable, units.PaicosQuantity):
            projection = units.PaicosQuantity(projection, variable_unit,
                                              a=self.snap.a, h=self.snap.h)

        return projection/area_per_pixel


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from paicos import root_dir
    import paicos as pa

    for use_units in [False, True]:

        pa.use_units(use_units)

        snap = pa.Snapshot(root_dir + '/data', 247)
        center = snap.Cat.Group['GroupPos'][0]
        if pa.units.enabled:
            R200c = snap.Cat.Group['Group_R_Crit200'][0].value
        else:
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
            projector = Projector(snap, center, widths, direction, npix=512)

            filename = root_dir + '/data/projection_{}_247.hdf5'.format(direction)
            image_file = pa.ArepoImage(filename, projector)

            Masses = projector.project_variable('0_Masses')
            print(Masses[0, 0])
            Volumes = projector.project_variable('0_Volumes')

            image_file.save_image('0_Masses', Masses)
            image_file.save_image('0_Volumes', Volumes)

            # snap.get_temperatures()
            TemperaturesTimesMasses = projector.project_variable(
                                    snap['0_Temperatures'] * snap['0_Masses'])
            image_file.save_image('TemperaturesTimesMasses', TemperaturesTimesMasses)

            # Move from temporary filename to final filename
            image_file.finalize()

            # Make a plot
            axes[ii].imshow(np.array((Masses/Volumes)), origin='lower',
                            extent=np.array(projector.extent), norm=LogNorm())
        plt.show()

        if not use_units:
            M = snap.converter.get_paicos_quantity(snap['0_Masses'], 'Masses')
            # Projection now has units
            projected_mass = projector.project_variable(M)
