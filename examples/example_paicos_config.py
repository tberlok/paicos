import paicos as pa

pa.use_units(True)
pa.use_only_user_functions(False)


def TemperaturesTimesMassesSquared(snap):
    return snap['0_Temperatures']*snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
