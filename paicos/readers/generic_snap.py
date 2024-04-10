import numpy as np
import astropy.units as u
import traceback
import numbers
from .paicos_readers import PaicosReader
from .. import settings
from .. import units as pu


class GenericSnapshot(PaicosReader):
    """
    This is a very simple Snapshot class, which
    was written for use with general data (e.g. if one
    wants to make images of Ramses simulations).

    Detailed documentation/example usage remains to be written,
    so please create an issue if you have trouble getting this to work.
    """

    def __init__(self, basedir='.', snapnum=None, basename="snap",
                 basesubdir='snapdir', load_all=True, to_physical=False,
                 only_init=False,
                 verbose=False):
        """
        Please use see the docstring for the method 'give_info'.
        """

        try:
            super().__init__(basedir, snapnum, basename, basesubdir,
                             load_all, to_physical, verbose)
            self.box = self.Header["BoxSize"]
            self.box_size = np.array([self.box, self.box, self.box])
            if settings.use_units:
                self.box_size = self.box_size * self.length
        except FileNotFoundError as e:
            # File not found, user will have to call give_info
            if not only_init:
                import warnings
                msg = ('\n\nset only_init=True to initialize without'
                       + ' loading a data file. You will then need'
                       + ' to call give_info instead\n\n')
                warnings.warn(msg)
                raise e

        self._auto_list = []

    def give_info(self, boxsize_in_code_units, time_in_code_units,
                  snapnum=None,
                  length_unit=None,
                  time_unit=None,
                  mass_unit=None,
                  comoving_sim=False,
                  hubble_param=1,
                  redshift=0,
                  Omega0=1,
                  OmegaLambda=0,
                  OmegaBaryon=1
                  ):

        """
        Parameters
        ----------

            snapnum : int
                The snapshot number that you are analyzing

            boxsize_in_code_units : float
                The box size of your simulation box, which is assumed to be cubic.


        Optional parameters (needed when using units are enabled)
        --------------------------

            time_in_code_units : float

            length_unit : astropy unit
                The code unit used in the simulation.
                For length e.g. 0.2 * u.Unit('kpc')

            time_unit : astropy unit
                The code unit used in the simulation.
                For time e.g. u.Unit('Myr')

            mass_unit : astropy unit
                The code unit used in the simulation.
                For mass e.g. u.Unit('Msun')
        """
        if hasattr(self, "Parameters"):
            err_msg = ("give_info should not be called"
                       + " if Parameters has already been"
                       + " via a previous call to give_info"
                       + " or by the PaicosReader base class")
            raise RuntimeError(err_msg)

        self.Parameters = {}
        self.Header = {}
        self.Config = {}

        self.snapnum = snapnum

        self.Header['Time'] = time_in_code_units
        self.Parameters['HubbleParam'] = hubble_param
        self.Header["BoxSize"] = boxsize_in_code_units

        if settings.use_units:
            err_msg = ('It is expected that you pass'
                       + ' this unit when pa.settings.use_units is True')

            assert length_unit is not None, err_msg
            assert time_unit is not None, err_msg
            assert mass_unit is not None, err_msg

            unit_length = 1.0 * length_unit
            unit_time = 1.0 * time_unit
            unit_mass = 1.0 * mass_unit
            assert isinstance(unit_length, u.Quantity)
            assert isinstance(unit_time, u.Quantity)
            assert isinstance(unit_mass, u.Quantity)

            unit_velocity = unit_length / unit_time

            self.Parameters['UnitLength_in_cm'] = unit_length.to('cm').value
            self.Parameters['UnitMass_in_g'] = unit_mass.to('g').value
            self.Parameters['UnitVelocity_in_cm_per_s'] = unit_velocity.to('cm/s').value

        if comoving_sim:
            self.Header['Redshift'] = redshift
            self.Parameters['ComovingIntegrationOn'] = 1
            self.Parameters['Omega0'] = Omega0
            self.Parameters['OmegaBaryon'] = OmegaBaryon
            self.Parameters['OmegaLambda'] = OmegaLambda
        else:
            self.Parameters['ComovingIntegrationOn'] = 0

        # Enable units
        self.get_units_and_other_parameters()
        if settings.use_units:
            self.enable_units()

        self.box = self.Header["BoxSize"]
        self.box_size = np.array([self.box, self.box, self.box])
        if settings.use_units:
            self.box_size = self.box_size * self.length

        if self.comoving_sim:
            raise RuntimeError("comoving_sim not tested - please create an issue!")

        if self._HubbleParam != 1:
            raise RuntimeError("h!=1 not tested - please create an issue!")

    def set_volumes(self, data, unit='arepo_length^3'):
        """
        Use this to set the cell volumes.
        """
        self.set_data(data, '0_Volume', unit)

    def set_positions(self, data, unit='arepo_length'):
        """
        Use this to set the cell positions.
        """
        assert data.shape[1] == 3
        self.set_data(data, '0_Coordinates', unit)

    def set_data(self, data, key, unit=None):
        """
        Set data in the snapshot

        Parameters
        ----------

            data : numpy array
                For instance the density, temperature etc

            key : str
                The dictionary key.

            unit : astropy unit
                You only need to set this if you have units turned on.
        """
        import warnings
        if settings.double_precision:
            # Load all variables with double precision
            if not issubclass(data.dtype.type, numbers.Integral):
                data = data.astype(np.float64)
        else:
            warnings.warn('\n\nThe cython routines expect double precision '
                          + 'and will fail unless settings.double_precision '
                          + 'is True.\n\n')
        if settings.use_units:
            assert unit is not None
            self[key] = pu.PaicosQuantity(data, unit, h=self._HubbleParam, a=self._Time,
                                          comoving_sim=self.comoving_sim)
        else:
            self[key] = data
