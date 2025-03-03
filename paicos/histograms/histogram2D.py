"""
This module defines a class for 2D histograms.
"""
import numpy as np
import h5py
from astropy import units as u
from .. import units as pu
from .. import util
from .. import settings
from .histogram import _make_bins


class Histogram2D:
    """
    This code defines a Histogram2D class which can be used to create 2D
    histograms. The class takes in the bin edges for the x and y axes, and an
    optional argument to indicate if the histogram should be in log scale. The
    class has methods to calculate the bin edges and centers, remove astro
    units, and create the histogram with a specific normalization. It also has
    a method to generate a color label for the histogram with units.
    """

    def __init__(self, snap, x, y, weights=None, bins_x=200, bins_y=200,
                 normalize=True, logscale=True):
        """
        Initialize the Histogram2D class with the bin edges for the x
        and y axes, and an optional argument to indicate if the
        histogram should be in log scale.

        Parameters:
            snap (Snapshot): the input snapshot

            x (array): The x data for the histogram

            y (array): The y data for the histogram

            weights (array): The weight data for the histogram, default
                             is None

            bins_x (tuple): Tuple of lower edge, upper edge and number
                            of bins for x axis. Alternatively an integer
                            denoting the number of bins spanning
                            x.min() to x.max().

            bins_y (tuple): Tuple of lower edge, upper edge and number
                            of bins for y axis. Alternatively an integer.

            normalize (bool): Indicates whether the histogram should be
                               normalized, default is True

            logscale (bool): Indicates whether to use logscale for the
                             histogram, default is True.


        """

        self.snap = snap

        if isinstance(x, str):
            self.x = snap[x]
        else:
            self.x = x

        if isinstance(y, str):
            self.y = snap[y]
        else:
            self.y = y

        if isinstance(weights, str):
            self.weights = snap[weights]
        else:
            self.weights = weights

        self.normalize = normalize

        if isinstance(bins_x, int):
            bins_x = [self.x.min(), self.x.max(), bins_x]

            if logscale:
                bins_x[0] = self.x[self.x > 0].min()
        else:
            assert bins_x[0] < bins_x[1], 'min and max values swapped!'

        if isinstance(bins_y, int):
            bins_y = [self.y.min(), self.y.max(), bins_y]

            if logscale:
                bins_y[0] = self.y[self.y > 0].min()
        else:
            assert bins_y[0] < bins_y[1], 'min and max values swapped!'

        self.logscale = logscale

        if logscale:
            assert bins_x[0] > 0
            assert bins_y[0] > 0

        self.edges_x, self.centers_x = _make_bins(bins_x, self.logscale)
        self.edges_y, self.centers_y = _make_bins(bins_y, self.logscale)
        self.lower_x = self.edges_x[0]
        self.lower_y = self.edges_y[0]
        self.upper_x = self.edges_x[-1]
        self.upper_y = self.edges_y[-1]

        self.extent = [self.lower_x, self.upper_x, self.lower_y, self.upper_y]

        self._get_image_properties()

        # Make the histogram
        self.hist2d = self._make_histogram()

        self.colorlabel = None

    def _get_image_properties(self):
        if settings.use_units:
            x_unit = self.edges_x.unit_quantity
            y_unit = self.edges_y.unit_quantity
            edges_x = self.edges_x.value
            edges_y = self.edges_y.value
            centers_x = self.centers_x.value
            centers_y = self.centers_y.value
        else:
            edges_x = self.edges_x
            edges_y = self.edges_y
            centers_x = self.centers_x
            centers_y = self.centers_y

        if self.logscale:
            dx = np.diff(np.log10(edges_x))
            dy = np.diff(np.log10(edges_y))
        else:
            dx = np.diff(edges_x)
            dy = np.diff(edges_y)

        dxx, dyy = np.meshgrid(dx, dy)

        if settings.use_units:
            dxx = dxx * x_unit
            dyy = dyy * y_unit

        self.area_per_bin = dxx * dyy

        if self.logscale and settings.use_units:
            self.area_per_bin *= u.Unit('dex')**(2) / self.area_per_bin.unit

        centers_x_mat, centers_y_mat = np.meshgrid(centers_x, centers_y)
        if settings.use_units:
            self.centers_x_mat = centers_x_mat * x_unit
            self.centers_y_mat = centers_y_mat * y_unit
        else:
            self.centers_x_mat = centers_x_mat
            self.centers_y_mat = centers_y_mat

    def get_colorlabel(self, x_symbol, y_symbol, weight_symbol=None):
        """
        Method to generate a color label for the histogram with units

        Parameters:
            x_symbol (string): Symbol for x axis
            y_symbol (string): Symbol for y axis
            weight_symbol (string): Symbol for weight of the histogram,
                                    default is None

        Returns:
            string: The color label for the histogram with units
        """

        assert settings.use_units

        unit_label = self.hist2d.label()[1:-1]

        if self.logscale:
            colorlabel = (r'/\left(\mathrm{d}\mathrm{log}_{10} ' + x_symbol
                          + r'\,\mathrm{d}\mathrm{log}_{10}'
                          + y_symbol + r'\right)'
                          + r'\;' + unit_label)
        else:
            colorlabel = (r'/\left(\mathrm{d}' + x_symbol
                          + r'\,\mathrm{d}' + y_symbol + r'\right)'
                          + r'\;' + unit_label)

        if weight_symbol is not None:
            if self.normalize:
                colorlabel = weight_symbol + \
                    r'_\mathrm{tot}^{-1}\,\mathrm{d}' + \
                    weight_symbol + colorlabel
            else:
                colorlabel = r'\mathrm{d}' + weight_symbol + colorlabel
        else:
            if self.normalize:
                colorlabel = r'\mathrm{pixel count}/\mathrm{total count}' + colorlabel
            else:
                colorlabel = r'\mathrm{pixel count}' + colorlabel
        self.colorlabel = r'$' + colorlabel + r'$'
        return self.colorlabel

    @util.remove_astro_units
    def _cython_make_histogram(self, x, y, edges_x, edges_y, weights):
        """
        Private method for making the 2D histogram using cython code
        """

        if settings.openMP_has_issues:
            from ..cython.histogram import get_hist2d_from_weights as get_hist2d
        else:
            from ..cython.histogram import get_hist2d_from_weights_omp as get_hist2d

        nbins_x = edges_x.shape[0] - 1
        nbins_y = edges_y.shape[0] - 1

        lower_x = edges_x[0]
        upper_x = edges_x[-1]
        lower_y = edges_y[0]
        upper_y = edges_y[-1]

        hist2d = get_hist2d(
            x, y, weights,
            lower_x, upper_x, nbins_x,
            lower_y, upper_y, nbins_y,
            self.logscale,
            numthreads=settings.numthreads_reduction)

        return hist2d

    def _make_histogram(self):
        """
        Private method to create the 2D histogram.

        Returns:
            hist2d (2D array): The 2D histogram
        """

        x = self.x
        y = self.y
        weights = self.weights
        normalize = self.normalize

        # Figure out units for the histogram
        if settings.use_units:
            if not normalize and (weights is not None):
                hist_units = weights.unit
            else:
                hist_units = u.Unit('')
            if self.logscale:
                hist_units *= u.Unit('dex')**(-2)
            else:
                hist_units /= x.unit * y.unit

            self.hist_units = hist_units

            assert x.unit == self.edges_x.unit
            assert y.unit == self.edges_y.unit

        if weights is None:
            # np.ones_like returns something with units,
            # should replace with np.ones and the shape
            weights = np.ones(x.shape, dtype=np.float64)

        hist2d = self._cython_make_histogram(x, y, self.edges_x,
                                             self.edges_y, weights)

        hist2d = hist2d.T

        if settings.use_units:
            hist2d = pu.PaicosQuantity(hist2d, self.hist_units, a=self.x._a,
                                       h=self.x._h,
                                       comoving_sim=self.x.comoving_sim)
        if normalize:
            norm = np.sum(hist2d.flatten()) * self.area_per_bin
            hist2d /= norm

            sanity = np.sum(self.area_per_bin * hist2d)
            if settings.use_units:
                np.testing.assert_allclose(sanity.value, 1.0)
                assert sanity.unit == u.Unit(''), f'{sanity.unit} should be dimensionless'
            else:
                np.testing.assert_allclose(sanity, 1.0)

        return hist2d

    def save(self, basedir, basename="2d_histogram"):
        """
        Saves the 2D histogram in the basedir directory.

        Parameters:
            basedir (path): The directory where the histogram should be saved.
            basename (string): The basename for the filename, which will take
                               the form::

                                   filename =  basename + f'_{snapnum:03d}.hdf5'
        """

        if basedir[-1] != '/':
            basedir += '/'

        snapnum = self.snap.snapnum
        filename = basedir + basename + f'_{snapnum:03d}.hdf5'
        with h5py.File(filename, 'w') as hdf5file:
            #
            hdf5file.create_group('hist_info')
            hdf5file['hist_info'].attrs['logscale'] = self.logscale
            hdf5file['hist_info'].attrs['normalize'] = self.normalize

            util.save_dataset(hdf5file, 'centers_x', self.centers_x)
            util.save_dataset(hdf5file, 'centers_y', self.centers_y)
            data = self.hist2d
            name = 'hist2d'
            attrs = {}
            if hasattr(data, 'unit'):
                hdf5file.create_dataset(name, data=data.value)
                attrs.update({'unit': self.hist2d.unit.to_string()})
            else:
                hdf5file.create_dataset(name, data=data)
            if self.colorlabel is not None:
                attrs.update({'colorlabel': self.colorlabel})
            else:
                print(("Unable to save colorlabel, please call "
                      + "the 'get_colorlabel' method before saving."))
            # Add attributes
            for key, attr in attrs.items():
                hdf5file[name].attrs[key] = attr

        util._copy_over_snapshot_information(self.snap, filename)
