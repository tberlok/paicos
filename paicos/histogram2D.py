import numpy as np
from .util import remove_astro_units


class Histogram2D:
    """
    This code defines a Histogram2D class which can be used to create 2D
    histograms. The class takes in the bin edges for the x and y axes, and an
    optional argument to indicate if the histogram should be in log scale. The
    class has methods to calculate the bin edges and centers, remove astro
    units, and create the histogram with a specific normalization. It also has
    a method to generate a color label for the histogram with units.
    """

    def __init__(self, bins_x, bins_y, logscale=True):
        """
        Initialize the Histogram2D class with the bin edges for the x
        and y axes, and an optional argument to indicate if the
        histogram should be in log scale.

        Parameters:
            bins_x (tuple): Tuple of lower edge, upper edge and number
                            of bins for x axis
            bins_y (tuple): Tuple of lower edge, upper edge and number
                            of bins for y axis
            logscale (bool): Indicates whether to use logscale for the
                             histogram, default is True.

        """

        self.logscale = logscale

        self.edges_x, self.centers_x = self._make_bins(bins_x)
        self.edges_y, self.centers_y = self._make_bins(bins_y)
        self.lower_x = self.edges_x[0]
        self.lower_y = self.edges_y[0]
        self.upper_x = self.edges_x[-1]
        self.upper_y = self.edges_y[-1]

        self.extent = [self.lower_x, self.upper_x, self.lower_y, self.upper_y]

    def _make_bins(self, bins):
        """
        Private method to calculate the edges and centers of the bins
        given lower, upper and number of bins.

        Parameters:
            bins (tuple): Tuple of lower edge, upper edge and number of
                          bins for the axis
        Returns:
            edges (array): Edges of the bins
            centers (array): Centers of the bins
        """

        lower, upper, nbins = bins
        edges, centers = self.__make_bins(lower, upper, nbins)
        from . import settings
        if settings.use_units:
            assert lower.unit == upper.unit
            edges = np.array(edges)*lower.unit_quantity
            centers = np.array(centers)*lower.unit_quantity
        return edges, centers

    @remove_astro_units
    def __make_bins(self, lower, upper, nbins):
        """
        Private method to calculate the edges and centers of the bins
        in logscale or linear scale based on the class variable logscale
        Parameters:
            lower (float): Lower edge of the bin
            upper (float): Upper edge of the bin
            nbins (int): Number of bins for the axis
        Returns:
            edges (array): Edges of the bins
            centers (array): Centers of the bins
        """

        if self.logscale:
            lower = np.log10(lower)
            upper = np.log10(upper)

        edges = lower + np.arange(nbins+1)*(upper-lower)/nbins
        centers = 0.5*(edges[1:] + edges[:-1])

        if self.logscale:
            edges = 10**edges
            centers = 10**centers

        return edges, centers

    def _find_norm(self, hist2d):
        """
        Private method to find the normalizing constant for the histogram
        Parameters:
            hist2d (2D array): The 2D histogram for which the normalizing
                               constant is needed
        Returns:
            norm (float): The normalizing constant
        """
        from .cython.histogram import find_normalizing_norm_of_2d_hist
        norm = find_normalizing_norm_of_2d_hist(hist2d, self.edges_x,
                                                self.edges_y)
        return norm

    def get_colorlabel(self, x_symbol, y_symbol, weight_symbol=None):
        """
        Method to generate a color label for the histogram with units
        Parameters:
            x_symbol (string): Symbol for x axis
            y_symbol (string): Symbol for y axis
            weight_symbol (string): Symbol for weight of the histogram,
                                    default is None
        Returns:
            colorlabel (string): The color label for the histogram with
                                 units
        """
        from . import settings

        assert settings.use_units

        unit_label = self.hist_units.to_string(format='latex')[1:-1]
        unit_label = r'[' + unit_label + r']'

        if self.logscale:
            colorlabel = (r'/\left(\mathrm{d}\mathrm{log}_{10} ' + x_symbol +
                          r'\,\mathrm{d}\mathrm{log}_{10}' +
                          y_symbol + r'\right)'
                          + r'\;' + unit_label)
        else:
            colorlabel = (r'/\left(\mathrm{d}' + x_symbol +
                          r'\,\mathrm{d}' + y_symbol + r'\right)'
                          + r'\;' + unit_label)

        if weight_symbol is not None:
            if self.normalized:
                colorlabel = weight_symbol + \
                    r'_\mathrm{tot}^{-1}\,\mathrm{d}' + \
                    weight_symbol + colorlabel
            else:
                colorlabel = r'\mathrm{d}' + weight_symbol + colorlabel
        else:
            if self.normalized:
                colorlabel = r'\mathrm{pixel count}/\mathrm{total count}' + colorlabel
            else:
                colorlabel = r'\mathrm{pixel count}' + colorlabel

        return (r'$' + colorlabel + r'$')

    def make_histogram(self, x, y, weights=None, normalize=True):
        """
        Method to create the 2D histogram given the x, y and weight data.
        Parameters:
            x (array): The x data for the histogram
            y (array): The y data for the histogram
            weights (array): The weight data for the histogram, default
                             is None
            normalized (bool): Indicates whether the histogram should be
                               normalized, default is True
        Returns:
            hist2d (2D array): The 2D histogram
            norm (float): Normalizing constant for the histogram
        """
        from .cython.histogram import get_hist2d_from_weights
        from . import settings
        from astropy import units as u

        self.normalized = normalize

        # Figure out units for the histogram
        if settings.use_units:
            if not normalize and (weights is not None):
                hist_units = weights.unit
            else:
                hist_units = u.Unit('')
            if self.logscale:
                hist_units *= u.Unit('dex')**(-2)
            else:
                hist_units /= x.unit*y.unit

            self.hist_units = hist_units

        if settings.use_units:
            assert x.unit == self.edges_x.unit
            assert y.unit == self.edges_y.unit

        if weights is None:
            weights = np.ones_like(x, dtype=np.float64)
        else:
            if settings.use_units:
                weights = np.array(weights.value, dtype=np.float64)
            else:
                weights = np.array(weights, dtype=np.float64)

        nbins_x = self.edges_x.shape[0] - 1
        nbins_y = self.edges_y.shape[0] - 1

        if settings.use_units:
            lower_x = self.edges_x[0].value
            upper_x = self.edges_x[-1].value
            lower_y = self.edges_y[0].value
            upper_y = self.edges_y[-1].value
            x = np.array(x.value, dtype=np.float64)
            y = np.array(y.value, dtype=np.float64)
        else:
            lower_x = self.edges_x[0]
            upper_x = self.edges_x[-1]
            lower_y = self.edges_y[0]
            upper_y = self.edges_y[-1]
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)

        hist2d = get_hist2d_from_weights(
            x, y, weights,
            lower_x, upper_x, nbins_x,
            lower_y, upper_y, nbins_y,
            self.logscale,
            numthreads=1)

        if normalize:
            hist2d /= self._find_norm(hist2d)

        self.hist = hist2d.T

        # TODO: Add units to returned object? Probably yes

        return hist2d.T
