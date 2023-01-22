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


        Example usage given below:

        import paicos as pa
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from paicos import root_dir

        pa.use_units(True)

        snap = pa.Snapshot(root_dir + '/data', 247)
        center = snap.Cat.Group['GroupPos'][0]

        snap.load_data(0, 'Density')
        snap.load_data(0, 'Masses')
        snap.get_temperatures()
        snap.get_volumes()

        T = snap.P['0_Temperatures']
        if pa.settings.use_units:
            rho = snap.P['0_Density'].to_physical.astro
            M = snap.P['0_Masses'].to_physical.astro
            V = snap.P['0_Volumes'].to_physical.astro
        else:
            rho = snap.P['0_Density']
            M = snap.P['0_Masses']
            V = snap.P['0_Volumes']

        # Set up bins
        bins_T = [T.min(), T.max()/10, 200]
        bins_rho = [rho.min(), rho.max()*1e-4, 300]

        # Create histogram object
        rhoT = pa.Histogram2D(bins_rho, bins_T, logscale=True)

        # Make 2D histogram
        hist = rhoT.make_histogram(rho, T, weights=M, normalize=True)

        plt.figure(1)
        plt.clf()

        if pa.settings.use_units:
            plt.pcolormesh(rhoT.centers_x.value, rhoT.centers_y.value,
                           rhoT.hist, norm=LogNorm())
            plt.xlabel(rhoT.centers_x.label('\\rho'))
            plt.ylabel(rhoT.centers_y.label('T'))
        else:
            plt.pcolormesh(rhoT.centers_x, rhoT.centers_y, rhoT.hist,
                           norm=LogNorm())
        plt.title('paicos Histogram2D')
        if rhoT.logscale:
            plt.xscale('log')
            plt.yscale('log')
        cbar = plt.colorbar()
        cbar.set_label(rhoT.get_colorlabel(r'\rho', 'T', 'M'))
        plt.title('hist2d paicos, pcolormesh')
        plt.show()

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
        from paicos import find_normalizing_norm_of_2d_hist
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
        from paicos import settings

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
        from paicos import get_hist2d_from_weights
        from paicos import settings
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logscale = True

    # Generate some fake data...
    Np = 10**6
    np.random.seed(30)
    x = 0.5 * np.random.randn(Np) + 11.5
    y = 2 * np.random.randn(Np) + 10.5
    weights = np.ones_like(x)

    bins_x = (x.min(), x.max(), 50)
    bins_y = (y.min(), y.max(), 150)

    # # Make histogram with matplotlib
    plt.figure(1)
    plt.clf()
    h, xedges, yedges, image_matplotlib = plt.hist2d(x, y, bins=[50, 150])
    plt.title('matplotlib hist2d')
    plt.colorbar()

    plt.figure(2)
    plt.clf()
    hist2d = Histogram2D(bins_x, bins_y, logscale)

    hist = hist2d.make_histogram(x, y, weights)
    # plt.imshow(hist, extent=hist2d.extent, aspect='auto', origin='lower')
    plt.pcolormesh(hist2d.centers_x, hist2d.centers_y, hist)
    plt.title('paicos Histogram2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    # # Make histogram with numpy
    plt.figure(3)
    plt.clf()
    pdf, xedges, yedges = np.histogram2d(x, y, weights=weights,
                                         bins=(hist2d.centers_x, hist2d.centers_y))
    plt.imshow(pdf.T, extent=hist2d.extent, aspect='auto', origin='lower')
    plt.title('hist2d numpy')
    plt.colorbar()

    # norm = ∑ᵢⱼ pdfᵢⱼ cell_areaᵢⱼ
    pdf_vec = np.reshape(pdf.T, np.product(pdf.shape))
    dxy_vec = np.reshape(np.outer(np.diff(yedges), np.diff(xedges)),
                         np.product(pdf.shape))
    norm = np.dot(pdf_vec, dxy_vec)
    pdf /= norm

    plt.figure(4)
    plt.clf()
    plt.pcolormesh(hist2d.centers_x, hist2d.centers_y, np.transpose(pdf))
    # plt.imshow(pdf.T, extent=hist2d.extent, aspect='auto', origin='lower')
    plt.title('hist2d numpy, normalized')
    plt.colorbar()

    plt.figure(5)
    plt.clf()
    plt.pcolormesh(hist2d.centers_x, hist2d.centers_y,
                   (np.transpose(pdf) - hist))
    # plt.imshow(pdf.T, extent=hist2d.extent, aspect='auto', origin='lower')
    plt.title('difference, Christoph and mine')
    plt.colorbar()

    plt.figure(6)
    plt.clf()
    plt.pcolormesh(hist2d.centers_x, hist2d.centers_y, hist)
    plt.title('paicos Histogram2D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar()
    # plt.imshow(pdf.T, extent=hist2d.extent, aspect='auto', origin='lower')
    plt.title('hist2d paicos, pcolormesh, log scale')

    plt.show()
