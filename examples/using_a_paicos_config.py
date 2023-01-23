import paicos as pa
import example_paicos_config

snap = pa.Snapshot(pa.root_dir + '/data', 247)

snap['0_TM2']

snap['1_Masses'].astro

# snap['0_Pressure'].astro
snap['0_PressureTimesVolumes'].astro
