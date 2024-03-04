import paicos as pa

pa.use_units(True)
pa.use_only_user_functions(False)


def TemperaturesTimesMassesSquared(snap, get_dependencies=False):
    if get_dependencies:
        return ['0_Temperatures', '0_Masses']
    return snap['0_Temperatures'] * snap['0_Masses']**2


def CosmicRaysTimesMasses(snap, get_dependencies=False):
    if get_dependencies:
        return ['0_CosmicRays', '0_Masses']
    return snap['0_CosmicRays'] * snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)
pa.add_user_function('0_CRM', CosmicRaysTimesMasses)


snap = pa.Snapshot(pa.data_dir, 247)


snap['0_TM2']
