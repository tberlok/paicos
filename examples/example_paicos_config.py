import paicos as pa

pa.use_units(True)
pa.use_only_user_functions(False)


def TemperaturesTimesMassesSquared(snap):
    return snap['0_Temperatures']*snap['0_Masses']**2


def MassesPartType1(snap):
    import numpy as np
    M = snap.masstable[1]*np.ones(snap['1_Coordinates'].shape[0],
                                  dtype=np.float32)
    return M


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
pa.add_user_function('1_Masses', MassesPartType1)
