from project_snapshot import make_projections
import load_partial_snap
import multiprocessing


snapnum = 247

simfolder = '/lustre/cosmo-plasm/zoom-simulations/halo_0004/adiabatic-mhd/zoom24/'

s = load_partial_snap.snapshot(simfolder + '/output', snapnum, 0)
nparts = s.Parameters['NumFilesPerSnapshot']

pool = multiprocessing.Pool(16)


def do_task(npart):
    make_projections(simfolder, snapnum, npart, 2048)
    return 0


output = pool.starmap(do_task, list(range(nparts)))
