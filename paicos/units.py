import astropy.units as u

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
gauss_B = (u.g/u.cm)**(0.5)/u.s
equiv_B = [(u.G, gauss_B, lambda x: x, lambda x: x)]
u.add_enabled_equivalencies(equiv_B)
