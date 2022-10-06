from project_snapshot import make_projections
import load_partial_snap
import multiprocessing


snapnum = 247

simfolder = '/lustre/cosmo-plasm/zoom-simulations/halo_0004/adiabatic-mhd/zoom24/'

s = load_partial_snap.snapshot(simfolder + '/output', snapnum, 0)
nparts = s.Parameters['NumFilesPerSnapshot']

pool = multiprocessing.Pool(16)

tasks = []
for npart in nparts:
    tasks.append([simfolder, snapnum, npart, 2048])

output = pool.starmap(make_projections, tasks)
