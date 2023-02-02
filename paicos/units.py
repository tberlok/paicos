import astropy.units as u
from astropy.units import Quantity
import numpy as np

_ns = globals()

small_a = u.def_unit(
    ["small_a"],
    prefixes=False,
    namespace=_ns,
    doc="Cosmological scale factor.",
    format={"latex": "a", "unicode": r"a"},
)
u.def_physical_type(small_a, "scale factor")


small_h = u.def_unit(
    ["small_h"],
    namespace=_ns,
    prefixes=False,
    doc='Reduced/"dimensionless" Hubble constant',
    format={"latex": r"h", "unicode": r"h"},
)

u.add_enabled_units(small_a)
u.add_enabled_units(small_h)

# Allows conversion between K and eV
u.add_enabled_equivalencies(u.temperature_energy())
# Allows conversion to Gauss (potential issues?)
# https://github.com/astropy/astropy/issues/7396
gauss_B = (u.g / u.cm)**(0.5) / u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]
scaling = small_a**(-2) * small_h
equiv_B_comoving = [(u.G * scaling, gauss_B * scaling, lambda x: x, lambda x: x)]
u.add_enabled_equivalencies(equiv_B)
u.add_enabled_equivalencies(equiv_B_comoving)


class PaicosQuantity(Quantity):

    """
    PaicosQuantity is a subclass of the astropy Quantity class which
    represents a number with some associated unit.

    This subclass in addition includes a and h factors used in the definition
    of comoving variables.

    Parameters
    ----------

    value: the numeric values of your data (similar to astropy Quantity)

    a: the cosmological scale factor of your data

    h: the reduced Hubble parameter, e.g. h = 0.7

    unit: a string, e.g. 'g/cm^3 small_a^-3 small_h^2' or astropy Unit
    The latter can be defined like this:

    from paicos import units as pu
    from astropy import units as u
    unit = u.g*u.cm**(-3)*small_a**(-3)*small_h**(2)

    The naming of small_a and small_h is to avoid conflict with the already
    existing 'annum' (i.e. a year) and 'h' (hour) units.

    Returns
    ----------

    Methods/properties
    ----------

    no_small_h: returns a new comoving quantity where the h-factors have
               been removed and the numeric value adjusted accordingly.

    to_physical: returns a new  object where both a and h factors have been
                 removed, i.e. we have switched from comoving values to
                 the physical value.

    label: Return a Latex label for use in plots.

    Examples
    ----------

    units = 'g cm^-3 small_a^-3 small_h^2'
    A = PaicosQuantity(2, units, h=0.7, a=1/128)

    # Create a new comoving quantity where the h-factors have been removed
    B = A.no_small_h

    # Create a new quantity where both a and h factor have been removed,
    # i.e. we have switched from a comoving quantity to the physical value

    C = A.to_physical

    """

    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0, h=None, a=None, comoving_sim=None):

        assert h is not None, 'Paicos quantity is missing a value for h'
        assert a is not None, 'Paicos quantity is missing a value for a'
        assert comoving_sim is not None, 'is this from a comoving_sim?'

        if hasattr(value, 'unit'):
            unit = value.unit
            value = value.value

        obj = super().__new__(cls, value, unit=unit, dtype=dtype, copy=copy,
                              order=order, subok=subok, ndmin=ndmin)

        obj._h = h
        obj._a = a
        obj._comoving_sim = comoving_sim

        return obj

    def __array_finalize__(self, obj):
        """
        Heavily inspired by the astropy Quantity version
        """
        super_array_finalize = super().__array_finalize__
        if super_array_finalize is not None:
            super_array_finalize(obj)

        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        # Set Paicos specific parameters
        self._h = getattr(obj, '_h', None)
        self._a = getattr(obj, '_a', None)
        self._comoving_sim = getattr(obj, '_comoving_sim', None)

    @property
    def a(self):
        """
        The scale factor.
        """
        if self.comoving_sim:
            return self._a
        else:
            raise RuntimeError('Non-comoving object has no scale factor')

    @property
    def h(self):
        """
        The reduced Hubble parameter
        """
        return self._h

    @property
    def comoving_sim(self):
        """
        Whether the simulation had ComovingIntegrationOn
        """
        return self._comoving_sim

    @property
    def z(self):
        """
        The redshift.
        """
        if self.comoving_sim:
            return 1. / self._a - 1.
        else:
            raise RuntimeError('Non-comoving object has no redshift')

    def lookback_time(self, reader_object):
        if self.comoving_sim:
            return reader_object.get_lookback_time(self.z)
        else:
            msg = 'lookback_time not defined for non-comoving sim'
            raise RuntimeError(msg)

    def age(self, reader_object):
        if self.comoving_sim:
            return reader_object.get_age(self.z)
        else:
            msg = 'age not defined for non-comoving sim'
            raise RuntimeError(msg)

    @property
    def time(self):
        if self.comoving_sim:
            msg = 'time not defined for comoving sim'
            raise RuntimeError(msg)
        else:
            from astropy import units as u
            return self._a * u.Unit('arepo_time')

    def __get_unit_dictionaries(self):
        codic = {}
        dic = {}
        for unit, power in zip(self.unit.bases, self.unit.powers):
            if unit == small_a or unit == small_h:
                codic[unit] = power
            else:
                dic[unit] = power
        for key in [small_h, small_a]:
            if key not in codic:
                codic[key] = 0

        return codic, dic

    @property
    def unit_quantity(self):
        """
        Returns a new PaicosQuantity with the same units as the current
        PaicosQuantity and a numeric value of 1.
        """
        return PaicosQuantity(1., self.unit, a=self._a, h=self.h,
                              comoving_sim=self.comoving_sim)

    @property
    def uq(self):
        """
        A short hand for the 'unit_quantity' method.
        """
        return self.unit_quantity

    def _construct_unit_from_dic(self, dic):
        """
        Construct unit from a dictionary with the format returned
        from __get_unit_dictionaries
        """
        return np.product([unit**dic[unit] for unit in dic])

    def _get_new_units(self, remove_list=[]):
        """
        Return units where base_strings have been modified.
        """
        unit_list = []
        for unit, power in zip(self.unit.bases, self.unit.powers):
            if unit not in remove_list:
                unit_list.append(unit**power)

        return np.product(unit_list)

    @property
    def separate_units(self):
        codic, dic = self.__get_unit_dictionaries()
        u_unit = self._construct_unit_from_dic(dic)
        pu_unit = self._construct_unit_from_dic(codic)
        return u_unit, pu_unit

    @property
    def hdf5_attrs(self):
        """
        Give the units as a dictionary for hdf5 data set attributes
        """
        return {'unit': self.unit.to_string()}

    @property
    def no_small_h(self):
        """
        Remove scaling with h, returning a quantity with adjusted values.
        """
        codic, dic = self.__get_unit_dictionaries()
        factor = self.h**codic[small_h]

        value = self.view(np.ndarray)
        new_unit = self._get_new_units([small_h])
        return self._new_view(value * factor, new_unit)

    @property
    def cgs(self):
        """
        Returns a copy of the current `PaicosQuantity` instance with CGS units.
        The value of the resulting object will be scaled.
        """
        u_unit, pu_unit = self.separate_units
        cgs_unit = u_unit.cgs
        new_unit = pu_unit * cgs_unit / cgs_unit.scale
        return self._new_view(self.value * cgs_unit.scale, new_unit)

    @property
    def si(self):
        """
        Returns a copy of the current `PaicosQuantity` instance with SI units.
        The value of the resulting object will be scaled.
        """
        u_unit, pu_unit = self.separate_units
        si_unit = u_unit.si
        new_unit = pu_unit * si_unit / si_unit.scale
        return self._new_view(self.value * si_unit.scale, new_unit)

    def to(self, unit, equivalencies=[], copy=True):
        """
        Convert to different units. Similar functionality to the astropy
        Quantity.to() method.
        """
        if isinstance(unit, str):
            unit = u.Unit(unit)

        # Fix so that it works regardless
        # of whether the pu_unit is included or not
        if (small_a in unit.bases) or (small_h in unit.bases):
            return super().to(unit, equivalencies, copy)
        else:
            _, pu_unit = self.separate_units
            return super().to(unit * pu_unit, equivalencies, copy)

    @property
    def arepo(self):
        """
        Return quantity in Arepo code units.
        """
        from astropy.units import UnitConversionError
        arepo_bases = set([u.Unit('arepo_mass'),
                           u.Unit('arepo_length'),
                           u.Unit('arepo_time')])
        try:
            return self.decompose(bases=arepo_bases)
        except UnitConversionError as inst:
            err_msg = ('Conversion to arepo_units does not work well for '
                       + 'Temperature and magnetic field strength in Gauss. '
                       + 'Astropy throws the following error: ' + str(inst))

            raise UnitConversionError(err_msg)

        return None

    @property
    def astro(self):
        """
        Return quantity in typical units used in cosmological simulations
        """
        return self.decompose(bases=[u.kpc, u.Msun, u.s, u.uG, u.keV, u.K])

    def decompose(self, bases=[]):
        """
        Decompose into a different set of units, e.g.

        A = B.decompose(bases=[u.kpc, u.Msun, u.s, u.uG, u.keV])

        small_a and small_h are automatically included in the bases.
        """
        u_unit, pu_unit = self.separate_units
        if len(bases) == 0 or pu_unit == u.Unit(''):
            return super().decompose(bases)
        else:
            if isinstance(bases, set):
                bases = list(bases)
            bases.append(small_a)
            bases.append(small_h)
            bases = set(bases)
            return super().decompose(bases)

    def label(self, variable=''):
        """
        Return a Latex string for use in plots. The optional
        input variable could be the Latex symbol for the physical variable,
        for instance \rho or \nabla\times\vec{v}.
        """

        a_sc, a_sc_str = self.__scaling_and_scaling_str(small_a)
        h_sc, h_sc_str = self.__scaling_and_scaling_str(small_h)

        co_label = a_sc_str + h_sc_str

        normal_unit = self._get_new_units([small_h, small_a])

        # unit_label = normal_unit.to_string(format='latex')[1:-1]

        codic, dic = self.__get_unit_dictionaries()

        unit_label = ''
        for ii, key in enumerate(dic.keys()):
            _, unit_str = self.__scaling_and_scaling_str(key)
            unit_label += unit_str
            if ii < len(dic.keys()) - 1:
                unit_label += r'\;'

        label = (co_label + r'\; \left[' + unit_label + r'\right]')

        # Get ckpc, cMpc, ckpc/h and Mkpc/h as used in literature
        if normal_unit == 'kpc' or normal_unit == 'Mpc':
            if (a_sc == 0) or (a_sc == 1):
                if (h_sc == 0) or (h_sc == -1):
                    if a_sc == 1:
                        label = r'\mathrm{c}' + unit_label
                    elif a_sc == 0:
                        label = unit_label
                    if h_sc == -1:
                        label = label + r'\;h^{-1}'

            label = '[' + label + ']'

        if len(variable) > 0:
            label = variable + r'\;' + label
        label = '$' + label + '$'

        return label

    @property
    def to_physical(self):
        """
        Returns a copy of the current `PaicosQuantity` instance with the
        a and h factors removed, i.e. transform from comoving to physical.
        The value of the resulting object is scaled accordingly.
        """
        codic, dic = self.__get_unit_dictionaries()
        factor = self.h**codic[small_h]

        if self.comoving_sim:
            factor *= self.a**codic[small_a]

        value = self.view(np.ndarray)
        new_unit = self._get_new_units([small_a, small_h])
        return self._new_view(value * factor, new_unit)

    def __scaling_and_scaling_str(self, unit):
        """
        Helper function to create labels
        """
        from fractions import Fraction
        codic, dic = self.__get_unit_dictionaries()
        # print(unit, dic, codic)
        if unit in codic.keys():
            scaling = codic[unit]
        elif unit in dic.keys():
            scaling = dic[unit]
        else:
            raise RuntimeError('should not happen')

        if unit in codic.keys():
            base_string = unit.to_string(format='unicode')
        elif unit in dic.keys():
            base_string = unit.to_string(format='latex')[1:-1]
        scaling_str = str(Fraction(scaling).limit_denominator(10000))
        if scaling_str == '0':
            scaling_str = ''
        elif scaling_str == '1':
            scaling_str = base_string
        else:
            scaling_str = base_string + '^{' + scaling_str + '}'
        return scaling, scaling_str

    def __sanity_check(self, value):
        """
        Function for sanity-checking addition, subtraction, multiplication
        and division of quantities. They should all have same a and h.
        """
        err_msg = "Operation requires objects to have same a and h value."
        if isinstance(value, PaicosQuantity):
            if value._a != self._a:
                info = ' Obj1._a={}, Obj2._a={}'.format(self._a, value._a)
                raise RuntimeError(err_msg + info)
            if value.h != self.h:
                info = ' Obj1.h={}, Obj2.h={}'.format(self.h, value.h)
                raise RuntimeError(err_msg + info)

    def _repr_latex_(self):
        number_part = super()._repr_latex_().split('\\;')[0]
        _, pu_units = self.separate_units
        u_latex = (self.unit / pu_units).to_string(format='latex')[1:-1]
        pu_latex = pu_units.to_string(format='latex')[1:-1]

        if pu_units != u.Unit(''):
            modified = number_part + '\\;' + u_latex + \
                '\\times' + pu_latex + '$'
        else:
            modified = number_part + '\\;' + u_latex + '$'
        return modified

    def __add__(self, value):
        self.__sanity_check(value)
        return super().__add__(value)

    def __sub__(self, value):
        self.__sanity_check(value)
        return super().__sub__(value)

    def __mul__(self, value):
        self.__sanity_check(value)
        return super().__mul__(value)

    def __truediv__(self, value):
        self.__sanity_check(value)
        return super().__truediv__(value)


class PaicosTimeSeries(PaicosQuantity):

    """
    PaicosQuantity is a subclass of the astropy Quantity class which
    represents a number with some associated unit.

    This subclass in addition includes a and h factors used in the definition
    of comoving variables.

    Parameters
    ----------

    value: the numeric values of your data (similar to astropy Quantity)

    a: the cosmological scale factor of your data

    h: the reduced Hubble parameter, e.g. h = 0.7

    unit: a string, e.g. 'g/cm^3 small_a^-3 small_h^2' or astropy Unit
    The latter can be defined like this:

    from paicos import units as pu
    from astropy import units as u
    unit = u.g*u.cm**(-3)*small_a**(-3)*small_h**(2)

    The naming of small_a and small_h is to avoid conflict with the already
    existing 'annum' (i.e. a year) and 'h' (hour) units.

    Returns
    ----------

    Methods/properties
    ----------

    no_small_h: returns a new comoving quantity where the h-factors have
               been removed and the numeric value adjusted accordingly.

    to_physical: returns a new  object where both a and h factors have been
                 removed, i.e. we have switched from comoving values to
                 the physical value.

    label: Return a Latex label for use in plots.

    Examples
    ----------

    units = 'g cm^-3 small_a^-3 small_h^2'
    A = PaicosQuantity(2, units, h=0.7, a=1/128)

    # Create a new comoving quantity where the h-factors have been removed
    B = A.no_small_h

    # Create a new quantity where both a and h factor have been removed,
    # i.e. we have switched from a comoving quantity to the physical value

    C = A.to_physical

    """

    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0, h=None, a=None, comoving_sim=None):

        if isinstance(value, list):
            a = np.array([value[i]._a for i in range(value.shape[0])])
            h = value[0].h  # Could check that they are all the same...
            value = np.array(value)
        elif isinstance(value, np.ndarray):
            assert h is not None
            assert a is not None
            assert isinstance(a, np.ndarray)
            assert a.shape[0] == value.shape[0]
        else:
            raise RuntimeError('unexpected input for value:', value)

        msg = 'PaicosTimeSeries only supports 1D arrays'
        assert len(value.shape) == 1, msg

        obj = super().__new__(cls, value, unit=unit, dtype=dtype, copy=copy,
                              order=order, subok=subok, ndmin=ndmin, h=h, a=a,
                              comoving_sim=comoving_sim)

        obj._h = h
        obj._a = a
        obj._comoving_sim = comoving_sim

        return obj

    @property
    def hdf5_attrs(self):
        """
        Give the units as a dictionary for hdf5 data set attributes
        """
        return {'unit': self.unit.to_string(), 'Paicos': 'PaicosTimeSeries'}

    def __sanity_check(self, value):
        """
        Function for sanity-checking addition, subtraction, multiplication
        and division of quantities. They should all have same a and h.
        """
        err_msg = "Operation requires objects to have same a and h value.\n"
        if isinstance(value, PaicosQuantity):
            msg = ('operations combining PaicosQuantity and '
                   + 'PaicosTimeSeries is not allowed.')
            raise RuntimeError(msg)
        elif isinstance(value, PaicosTimeSeries):
            try:
                info = '\nObj1.a={}.\n\nObj2.a={}'.format(self._a, value._a)
                np.testing.assert_array_equal(value._a, self._a)
            except AssertionError:
                raise RuntimeError(err_msg + info)
            if value.h != self.h:
                info = '\nObj1.h={}.\n\nObj2.h={}'.format(self.h, value.h)
                raise RuntimeError(err_msg + info)
