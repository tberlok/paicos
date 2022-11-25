from astropy.units import Quantity
import numpy as np
import astropy.units as u
import units as pu


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
    unit = u.g*u.cm**(-3)*pu.small_a**(-3)*pu.small_h**(2)

    The naming of small_a and small_h is to avoid conflict with the already
    existing 'annum' (i.e. a year) and 'h' (hour) units.

    Returns
    ----------

    Methods/properties
    ----------

    no_smallh: returns a new comoving quantity where the h-factors have
               been removed and the numeric value adjusted accordingly.

    to_physical: returns a new  object where both a and h factors have been
                 removed, i.e. we have switched from comoving values to
                 the physical value.

    to_parameters(a=None, h=None): Returns a new object where the values of
                                   a and h have been changed.

    label: Return a Latex label for use in plots.

    Examples
    ----------

    units = 'g cm^-3 small_a^-3 small_h^2'
    A = PaicosQuantity(2, units, h=0.7, a=1/128)

    # Create a new comoving quantity where the h-factors have been removed
    B = A.no_smallh

    # Create a new quantity where both a and h factor have been removed,
    # i.e. we have switched from a comoving quantity to the physical value

    C = A.to_physical

    """

    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0, h=1, a=1):

        obj = super().__new__(cls, value, unit=unit, dtype=dtype, copy=copy,
                              order=order, subok=subok, ndmin=ndmin)

        obj.h = h
        obj.a = a

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.h = getattr(obj, 'h', None)
        self.a = getattr(obj, 'a', None)

    def _get_unit_dictionaries(self):

        codic = {}
        dic = {}

        for unit, power in zip(self.unit.bases, self.unit.powers):
            if unit == pu.small_a or unit == pu.small_h:
                codic[unit] = power
            else:
                dic[unit] = power
        for key in [pu.small_h, pu.small_a]:
            if key not in codic:
                codic[key] = 0

        return codic, dic

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
    def no_smallh(self):
        codic, dic = self._get_unit_dictionaries()
        factor = self.h**codic[pu.small_h]
        if factor == 1:
            return PaicosQuantity(self.value, self.unit, h=self.h, a=self.a)
        else:
            new_unit = self._get_new_units([pu.small_h])
            new = Quantity.__mul__(self, factor)

            return PaicosQuantity(new.value, new_unit, h=self.h, a=self.a)

    def label(self, variable=''):

        a_sc, a_sc_str = self._scaling_and_scaling_str(pu.small_a)
        h_sc, h_sc_str = self._scaling_and_scaling_str(pu.small_h)

        co_label = a_sc_str + h_sc_str

        normal_unit = self._get_new_units([pu.small_h, pu.small_a])

        unit_label = normal_unit.to_string(format='latex')[1:-1]

        label = ('$' + variable + r'\;' + co_label + r'\; \left[' +
                 unit_label + r'\right]$')

        # Get ckpc, cMpc, ckpc/h and Mkpc/h as used in literature
        if normal_unit == 'kpc' or normal_unit == 'Mpc':
            length_label = unit_label[1:-1]
            if a_sc == 1:
                label = r'$\mathrm{c}' + length_label + '$'

            if h_sc == -1:
                label = '$' + length_label + r'/h$'

        return label

    @property
    def to_physical(self):
        codic, dic = self._get_unit_dictionaries()
        factor = self.h**codic[pu.small_h]
        factor *= self.a**codic[pu.small_a]

        if factor == 1:
            return PaicosQuantity(self.value, self.unit, h=self.h, a=self.a)
        else:
            new_unit = self._get_new_units([pu.small_a, pu.small_h])

            new = Quantity.__mul__(self, factor)

            return PaicosQuantity(new.value, new_unit, h=self.h, a=self.a)

    def to_comoving(self, a_and_h_scaling):

        raise RuntimeError('Not implemented')

    def to_arepo_units(self):
        pass

    def _scaling_and_scaling_str(self, unit):
        from fractions import Fraction
        codic, dic = self._get_unit_dictionaries()
        scaling = codic[unit]
        base_string = unit.to_string(format='unicode')
        scaling_str = str(Fraction(scaling).limit_denominator(10000))
        if scaling_str == '0':
            scaling_str = ''
        elif scaling_str == '1':
            scaling_str = base_string
        else:
            scaling_str = base_string + '^{' + scaling_str + '}'
        return scaling, scaling_str

    def __getitem__(self, key):
        """
        Comoving quantity loses the astropy units when creating a slice.
        This fixes that issue.
        """

        out = super().__getitem__(key)
        new = PaicosQuantity(out.value, self.unit, h=self.h, a=self.a,
                             copy=False)
        return new


if __name__ == '__main__':
    A = PaicosQuantity(1, pu.small_a**2*pu.small_h*u.cm**4, h=0.7, a=1/128)
    C = PaicosQuantity(np.ones((4, 4))*2.1,
                       'g cm^-3 small_a^-3 small_h^2', h=0.7, a=1/128)

    # Initialize 10 μG field at a = 1
    B = PaicosQuantity(10, 'uG small_a^-2 small_h', h=1, a=1)
