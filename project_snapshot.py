import load_partial_snap
from arepo_image import ArepoImage
from sph_like_projection import Projector
import os


def make_projections(simfolder, snapnum, partfile=None):
    if simfolder[-1] != '/':
        simfolder += '/'

    snap = load_partial_snap.snapshot(simfolder + 'output/', snapnum, partfile)
    center = snap.Cat.Group['GroupPos'][0]
    R200c = snap.Cat.Group['Group_R_Crit200'][0]

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
        [2*R200c, 10000, 10000],
        [10000, 2*R200c, 10000],
        [10000, 10000, 2*R200c],
        )

    width_vec = (
        [10000, 10000, 10000],
        [10000, 10000, 10000],
        [10000, 10000, 10000],
        )

    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        p = Projector(snap, center, widths, direction, npix=512)

        image_file = ArepoImage(proj_dir + proj_filename.format(direction, snapnum, partfile),
                                p.snap.filename, center, widths, direction)

        for variable in fields:
            image = p.project_variable(variable)
            image_file.save_image(variable, image)

        # Move from temporary filename to final filename
        image_file.finalize()


if __name__ == '__main__':
    make_projections('.', 247)
