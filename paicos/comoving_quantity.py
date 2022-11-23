import numpy as np
from astropy.units import Quantity
from astropy.units import format as unit_format


class ComovingQuantity(Quantity):

    """
    We make a subclass of the astropy Quantity class in order to also
    include comoving variables.

    I've implemented most basic functionality, i.e. addition, subtraction,
    division, multiplication, powers, numpy sqrt and cbrt.
    Testing to follow!
    """

    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0, comoving_dic=None):

        assert 'a_scaling' in comoving_dic
        assert 'h_scaling' in comoving_dic
        assert 'small_h' in comoving_dic
        assert 'scale_factor' in comoving_dic

        obj = super().__new__(cls, value, unit=unit, dtype=dtype, copy=copy,
                              order=order, subok=subok, ndmin=ndmin)

        obj.comoving_dic = comoving_dic

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.comoving_dic = getattr(obj, 'comoving_dic', None)

    def no_small_h(self):
        factor = self.comoving_dic['small_h']**self.comoving_dic['h_scaling']
        new = Quantity.__mul__(self, factor)

        comoving_dic = {}
        for key in self.comoving_dic:
            comoving_dic.update({key: self.comoving_dic[key]})
        comoving_dic['h_scaling'] = 0
        new = ComovingQuantity(new.value, new.unit,
                               comoving_dic=comoving_dic)
        return new

    def get_label(self, variable=''):
        """
        Here we use information from self.unit to return labels for plots
        """
        from fractions import Fraction
        s = unit_format.Latex.to_string(self.unit)
        label = s[1:-1]

        a_sc = self.comoving_dic['a_scaling']
        h_sc = self.comoving_dic['h_scaling']

        a_sc_str = str(Fraction(a_sc).limit_denominator(10000))
        h_sc_str = str(Fraction(h_sc).limit_denominator(10000))
        if a_sc_str == '0':
            a_sc_str = ''
        elif a_sc_str == '1':
            a_sc_str = 'a'
        else:
            a_sc_str = 'a^{' + a_sc_str + '}'

        if h_sc_str == '0':
            h_sc_str = ''
        elif h_sc_str == '1':
            h_sc_str = 'h'
        else:
            h_sc_str = 'h^{' + h_sc_str + '}'

        co_label = a_sc_str + h_sc_str

        label = ('$' + variable + r'\;' + co_label + r'\; \left[' +
                 label + r'\right]$')

        if self.unit == 'kpc' or self.unit == 'Mpc':
            length_label = s[1:-1]
            if a_sc == 1:
                length_label = r'\mathrm{c}' + length_label
            else:
                length_label = r'a^{}'.format(a_sc) + length_label

            if h_sc == -1:
                length_label = length_label + r'/h'
            else:
                length_label = r'h^{}'.format(h_sc) + length_label
            label = '$' + length_label + '$'

        label = label.replace('a^1', 'a')
        label = label.replace('h^1', 'h')
        label = label.replace('a^0', '')
        label = label.replace('h^0', '')

        return label

    def get_attribute(self):

        attribute_dic = {}
        for key in self.comoving_dic:
            attribute_dic.update({key: self.comoving_dic[key]})
        attribute_dic.update({'units': self.unit.to_string()})

        return attribute_dic

    def to_physical(self):
        """
        Convert from comoving to physical values.
        """
        factor_h = self.comoving_dic['small_h']**self.comoving_dic['h_scaling']
        factor_a = self.comoving_dic['scale_factor']**self.comoving_dic['a_scaling']
        new = Quantity.__mul__(self, factor_h*factor_a)

        comoving_dic = {}
        for key in self.comoving_dic:
            comoving_dic.update({key: self.comoving_dic[key]})
        comoving_dic['h_scaling'] = 0
        comoving_dic['a_scaling'] = 0
        new = ComovingQuantity(new.value, new.unit,
                               comoving_dic=comoving_dic)
        return new

    def _repr_latex_(self):
        """
        Get nice presentation in Jupyter notebooks
        """
        s = Quantity._repr_latex_(self)
        label = s[1:-1]

        a_sc = self.comoving_dic['a_scaling']
        h_sc = self.comoving_dic['h_scaling']
        if a_sc == 0:
            comoving = False
        else:
            comoving = True
        if comoving:
            if h_sc != 0:
                if h_sc == 1 and a_sc == 1:
                    co_label = r'a h'.format(a_sc, h_sc)
                elif a_sc == 1:
                    co_label = r'a h^{}'.format(h_sc)
                elif h_sc == 1:
                    co_label = r'a h'.format(a_sc)
                else:
                    co_label = r'a^{} h^{}'.format(a_sc, h_sc)
            else:
                if a_sc == 1:
                    co_label = r'a'
                else:
                    co_label = r'a^{}'.format(a_sc)
        else:
            if h_sc != 0:
                if h_sc == 1:
                    co_label + r'h'
                else:
                    co_label + r'h^{}'.format(h_sc)
            else:
                co_label = ''
        if len(co_label) > 0:
            label = ('$' + label + r'\times ' + co_label + '$')
        else:
            label = s
        return label

    def __repr__(self):
        s = Quantity.__repr__(self)
        a_sc = self.comoving_dic['a_scaling']
        h_sc = self.comoving_dic['h_scaling']
        if a_sc == 0:
            comoving = False
        else:
            comoving = True
        s = s + '\n' + 'Comoving: {}'.format(comoving)
        if comoving:
            if h_sc != 0:
                s = s + '\nScaling: a^{} h^{}'.format(a_sc, h_sc)
            else:
                s = s + '\nScaling: a^{}'.format(a_sc)
        else:
            if h_sc != 0:
                s = s + '\nScaling: h^{}'.format(h_sc)
        return s

    def __add__(self, value):
        """
        """
        if isinstance(value, ComovingQuantity):
            # Check same scaling
            for key in self.comoving_dic.keys():
                assert self.comoving_dic[key] == value.comoving_dic[key]
        else:
            err_msg = ("ComovingQuantities can only be " +
                       "added to other comoving quantities")
            raise RuntimeError(err_msg)
        return Quantity.__add__(self, value)

    def __sub__(self, value):
        if isinstance(value, ComovingQuantity):
            # Check same scaling
            for key in self.comoving_dic.keys():
                assert self.comoving_dic[key] == value.comoving_dic[key]
        else:
            err_msg = ("ComovingQuantities can only be " +
                       "added to other comoving quantities")
            raise RuntimeError(err_msg)
        return Quantity.__sub__(self, value)

    def __mul__(self, value):
        new = Quantity.__mul__(self, value)
        if isinstance(value, ComovingQuantity):
            assert self.comoving_dic['small_h'] == value.comoving_dic['small_h']
            assert self.comoving_dic['scale_factor'] == value.comoving_dic['scale_factor']
            a_sc = self.comoving_dic['a_scaling']
            h_sc = self.comoving_dic['h_scaling']
            a_sc2 = value.comoving_dic['a_scaling']
            h_sc2 = value.comoving_dic['h_scaling']
            comoving_dic = {'a_scaling': a_sc + a_sc2,
                            'h_scaling': h_sc + h_sc2,
                            'small_h': self.comoving_dic['small_h'],
                            'scale_factor': self.comoving_dic['scale_factor']}
            new = ComovingQuantity(new.value, new.unit,
                                   comoving_dic=comoving_dic)
        return new

    def __truediv__(self, value):
        new = Quantity.__truediv__(self, value)
        if isinstance(value, ComovingQuantity):
            assert self.comoving_dic['small_h'] == value.comoving_dic['small_h']
            assert self.comoving_dic['scale_factor'] == value.comoving_dic['scale_factor']
            a_sc = self.comoving_dic['a_scaling']
            h_sc = self.comoving_dic['h_scaling']
            a_sc2 = value.comoving_dic['a_scaling']
            h_sc2 = value.comoving_dic['h_scaling']
            comoving_dic = {'a_scaling': a_sc - a_sc2,
                            'h_scaling': h_sc - h_sc2,
                            'small_h': self.comoving_dic['small_h'],
                            'scale_factor': self.comoving_dic['scale_factor']}
            new = ComovingQuantity(new.value, new.unit,
                                   comoving_dic=comoving_dic)

        return new

    def __pow__(self, value):
        new = Quantity.__pow__(self, value)
        a_sc = self.comoving_dic['a_scaling']
        h_sc = self.comoving_dic['h_scaling']
        comoving_dic = {'a_scaling': a_sc*value,
                        'h_scaling': h_sc*value,
                        'small_h': self.comoving_dic['small_h'],
                        'scale_factor': self.comoving_dic['scale_factor']}
        new = ComovingQuantity(new.value, new.unit,
                               comoving_dic=comoving_dic)
        return new

    def __array_ufunc__(self, function, method, *inputs, **kwargs):

        a_sc = self.comoving_dic['a_scaling']
        h_sc = self.comoving_dic['h_scaling']
        comoving_dic = self.comoving_dic
        if function == np.sqrt:
            comoving_dic = {'a_scaling': a_sc/2,
                            'h_scaling': h_sc/2}
        if function == np.square:
            comoving_dic = {'a_scaling': a_sc*2,
                            'h_scaling': h_sc*2}
        elif function == np.cbrt:
            comoving_dic = {'a_scaling': a_sc/3,
                            'h_scaling': h_sc/3}
        elif function == np.multiply:
            pass
            if isinstance(inputs[1], ComovingQuantity):
                a_sc2 = inputs[1].comoving_dic['a_scaling']
                h_sc2 = inputs[1].comoving_dic['h_scaling']
                comoving_dic = {'a_scaling': a_sc + a_sc2,
                                'h_scaling': h_sc + h_sc2}
        elif function == np.true_divide:
            pass
            if isinstance(inputs[1], ComovingQuantity):
                a_sc2 = inputs[1].comoving_dic['a_scaling']
                h_sc2 = inputs[1].comoving_dic['h_scaling']
                comoving_dic = {'a_scaling': a_sc - a_sc2,
                                'h_scaling': h_sc - h_sc2}
        elif function == np.add:
            pass
        elif function == np.subtract:
            pass
        elif function == np.power:
            comoving_dic = {'a_scaling': a_sc*inputs[1],
                            'h_scaling': h_sc*inputs[1]}
        else:
            raise RuntimeError('not implemented: {}'.format(function))

        new = super().__array_ufunc__(function, method, *inputs, **kwargs)

        comoving_dic.update({'small_h': self.comoving_dic['small_h'],
                            'scale_factor': self.comoving_dic['scale_factor']})
        new = ComovingQuantity(new.value, new.unit,
                               comoving_dic=comoving_dic)
        return new


if __name__ == '__main__':
    from astropy import units as u
    unit_length = 1.0*u.cm
    unit_mass = 1.0*u.g
    unit_velocity = 1.0*u.cm/u.s
    unit_time = unit_length / unit_velocity
    unit_energy = unit_mass * unit_velocity**2
    unit_pressure = (unit_mass/unit_length) / unit_time**2

    arepo_units = {'unit_length': unit_length,
                   'unit_mass': unit_mass,
                   'unit_velocity': unit_velocity,
                   'unit_time': unit_time,
                   'unit_energy': unit_energy,
                   'unit_pressure': unit_pressure}

    D = ComovingQuantity(1, unit=u.g*u.cm**(-3),
                         comoving_dic={'a_scaling': 2, 'h_scaling': 2,
                                       'small_h': 0.7, 'scale_factor': 0.1})
    E = ComovingQuantity(2, unit=u.g*u.cm**(-3), comoving_dic={'a_scaling': 2,
                                                               'h_scaling': 2,
                                                               'small_h': 0.7,
                                                               'scale_factor': 0.1})

    F = ComovingQuantity(2, unit=u.g*u.cm**(-3), comoving_dic={'a_scaling': 1,
                                                               'h_scaling': 2,
                                                               'small_h': 0.7,
                                                               'scale_factor': 0.1})

    b = np.random.rand(2, 2)
    G = ComovingQuantity(b, unit=u.g*u.cm**(-3), comoving_dic={'a_scaling': 1,
                                                               'h_scaling': 2,
                                                               'small_h': 0.7,
                                                               'scale_factor': 0.1})

    S = (F + G)**2/E*10.
