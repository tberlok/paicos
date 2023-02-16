import paicos as pa
from paicos import root_dir
pa.use_units(True)
snap = pa.Snapshot(root_dir + '/data', 247)

snap['0_MagneticField']
snap['0_Volume']
snap['0_Temperatures']

snap['0_Volume'].no_small_h.to('kpc3')
snap['0_Volume'].to('kpc3').no_small_h

snap['0_Volume'].to('kpc3 small_a^3 small_h^-2')
