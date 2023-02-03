import paicos as pa
from paicos import root_dir

pa.use_units(True)
pa.use_only_user_functions(False)


def TemperaturesTimesMassesSquared(snap, get_depencies=False):
    if get_depencies:
        return ['0_Temperatures', '0_Masses']
    return snap['0_Temperatures'] * snap['0_Masses']**2


def CosmicRaysTimesMasses(snap, get_depencies=False):
    if get_depencies:
        return ['0_CosmicRays', '0_Masses']
    return snap['0_CosmicRays'] * snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
pa.add_user_function('0_CRM', CosmicRaysTimesMasses)


snap = pa.Snapshot(root_dir + '/data', 247)


snap['0_TM2']
