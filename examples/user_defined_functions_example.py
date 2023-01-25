import paicos as pa
from paicos import root_dir

pa.use_units(True)

snap = pa.Snapshot(root_dir + '/data', 247)


def TemperaturesTimesMassesSquared(snap):
    return snap['0_Temperatures']*snap['0_Masses']**2


pa.add_user_function('0_TM2', TemperaturesTimesMassesSquared)

snap['0_TM2']

pa.use_only_user_functions(True)

# Should then fail
try:
    snap['0_Volume']
except RuntimeError:
    print('Expected failure achieved...')
