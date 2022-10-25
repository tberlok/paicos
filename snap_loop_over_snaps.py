from paicos import load_partial_snap
from paicos import ArepoImage
from paicos import Projector
import os


def make_projections(simfolder, snapnum, partfile=None, npix=512):
    if simfolder[-1] != '/':
        simfolder += '/'

    snap = load_partial_snap.snapshot(simfolder + 'output/', snapnum, partfile)
    center = snap.Cat.Group['GroupPos'][0]

    fields = ['Masses', 'Volume', 'TemperatureTimesMasses',
              'EnergyDissipation', 'MachnumberTimesEnergyDissipation',
              'MagneticFieldSquaredTimesVolume', 'PressureTimesVolume',
              'EnstrophyTimesMasses']

    if partfile is None:
        proj_dir = simfolder + 'projections/'
        proj_filename = 'projection_{}_{}.hdf5'
    else:
        proj_dir = simfolder + 'projections/projdir_{:03d}/'.format(snapnum)
        proj_filename = 'projection_{}_{}.{}.hdf5'

    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

    width_vec = (
        [10000, 10000, 10000],
        [10000, 10000, 10000],
        [10000, 10000, 10000],
        )

    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        p = Projector(snap, center, widths, direction, npix=npix)

        image_file = ArepoImage(proj_dir + proj_filename.format(direction, snapnum, partfile),
                                p.snap.filename, center, widths, direction)

        for variable in fields:
            image = p.project_variable(variable)
            image_file.save_image(variable, image)

        # Move from temporary filename to final filename
        image_file.finalize()


if __name__ == '__main__':
    import multiprocessing
    import h5py
    import sys
    import socket

    
    
    zoom = 4

    #halo_nums = [417, 261, 56, 41, 8, 420, 239, 104, 33, 19, 542, 260, 112, 22, 9, 344, 194, 98, 49, 20]
    # 2 not working, 112 just submitted
    #halo_nums = [4]
    halo_num = 112

    for snapnum in range(129, 248):
        print('working on snap, halo', snapnum, halo_num)
        simfolder = '/lustre/cosmo-plasm/zoom-simulations/halo_{:04d}/adiabatic-mhd/zoom{}/'.format(halo_num, zoom)
        if socket.gethostname() == 'nnewl3.newton21.nnew':
            simfolder = '/llust21/cosmo-plasm/zoom-simulations/halo_{:04d}/adiabatic-mhd/zoom{}/'.format(halo_num, zoom)

        s = load_partial_snap.snapshot(simfolder + '/output', snapnum)
        nparts = s.Parameters['NumFilesPerSnapshot']

        if nparts == 1:
            make_projections(simfolder, snapnum, None, npix=512)
        else:
            pool = multiprocessing.Pool(16)
            tasks = []
            for npart in range(nparts):
                tasks.append([simfolder, snapnum, npart, 512])

            output = pool.starmap(make_projections, tasks)

            proj_dir = simfolder + 'projections/projdir_{:03d}/'.format(snapnum)
            proj_filename = 'projection_{}_{}.{}.hdf5'
            tmp_new_filename = simfolder + 'projections/tmp_projection_{}_{}.hdf5'
            new_filename = simfolder + 'projections/projection_{}_{}.hdf5'

            for direction in ['x', 'y', 'z']:
                # Turn into a generic merge code!
                for npart in range(nparts):
                    print(npart)
                    filename = proj_dir + proj_filename.format(direction, snapnum, npart)
                    with h5py.File(filename, 'r') as g:
                        if npart == 0:
                            with h5py.File(tmp_new_filename.format(direction, snapnum), 'w') as f:
                                for name in g.keys():
                                    if type(g[name]) is h5py._hl.dataset.Dataset:
                                        f.create_dataset(name, data=g[name][...])
                                    elif type(g[name]) is h5py._hl.group.Group:
                                        group = name
                                        f.create_group(group)
                                        for key in g[group].attrs.keys():
                                            f[group].attrs[key] = g[group].attrs[key]
                        else:
                            with h5py.File(tmp_new_filename.format(direction, snapnum), 'r+') as f:
                                for name in g.keys():
                                    if type(g[name]) is h5py._hl.dataset.Dataset:
                                        f[name][...] += g[name][...]

                os.rename(tmp_new_filename.format(direction, snapnum),
                          new_filename.format(direction, snapnum))
