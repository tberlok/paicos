import multiprocessing
from paicos import load_partial_snap
import arepo_snap
import time
import numpy as np

folder = '/lustre/cosmo-plasm/zoom-simulations/halo_0004/adiabatic-mhd/zoom24/output/'

t = time.time()
procs = 16


def load_part(part):
    s2 = load_partial_snap.snapshot(folder, 247, part)
    s2.load_data(0, 'Coordinates')
    s2.load_data(0, 'Masses')
    return np.sum(s2.P['0_Masses'])


tasks = [[ii] for ii in range(176)]
with multiprocessing.Pool(procs) as process_pool:
    out = process_pool.starmap(load_part, tasks)
    print(np.sum(out))
print('took ', time.time()-t, ' seconds with ', procs, ' processors')

t = time.time()
s2 = arepo_snap.snapshot(folder, 247)
s2.load_data(0, 'Coordinates')
s2.load_data(0, 'Masses')
print(np.sum(s2.P['0_Masses']))
print('took ', time.time()-t, ' seconds with 1 processor loading all files')
