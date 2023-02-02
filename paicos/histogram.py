import numpy as np
from . import util
from . import settings


class Histogram:
    """
    This is a brief class for efficient computation of many 1D histograms
    using the same x-variables and binning but different weights.

    The heavy part of the computation is stored in idigit
    """

    def __init__(self, x, bins, logscale=False, verbose=False):
        """
        Initialize the class.

        Parameters:
            x_variable (array): x-coordinates of the data.

            bins (array): Bin edges for the histogram.

            verbose (bool, optional): If True, prints the time taken to
                                      digitize the x-coordinates.
        """

        if isinstance(bins, int):
            bins = [x.min(), x.max(), bins]

        self.logscale = logscale
        self.verbose = verbose

        self.edges, self.centers = self._make_bins(bins)
        self.bin_centers = self.centers
        self.bin_edges = self.edges

        if self.verbose:
            import time
            t = time.time()
            print('Digitize for histogram begun')

        self.idigit = np.digitize(x, self.edges)
        if self.verbose:
            print('This took {:1.2f} seconds'.format(time.time() - t))

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
            edges = np.array(edges) * lower.unit_quantity
            centers = np.array(centers) * lower.unit_quantity
        return edges, centers

    @util.remove_astro_units
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

        edges = lower + np.arange(nbins + 1) * (upper - lower) / nbins
        centers = 0.5 * (edges[1:] + edges[:-1])

        if self.logscale:
            edges = 10**edges
            centers = 10**centers

        return edges, centers

    def hist(self, weights):
        """
        Compute the histogram of the data.

        Parameters:
            weights (array): Weights of the data.

        Returns:
            array: Histogram of the data.
        """
        from .cython.histogram import get_hist_from_weights_and_idigit as func

        get_hist_from_weights_and_idigit = util.remove_astro_units(func)

        # compute histogram using pre-digitized x-coordinates and given weights
        hist = get_hist_from_weights_and_idigit(self.edges.shape[0], weights,
                                                self.idigit)
        if settings.use_units:
            hist = hist * weights.unit_quantity

        return hist
