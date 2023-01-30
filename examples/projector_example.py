import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from paicos import root_dir
import paicos as pa
import numpy as np

for use_units in [False, True]:

    pa.use_units(use_units)

    snap = pa.Snapshot(root_dir + '/data', 247)
    center = snap.Cat.Group['GroupPos'][0]
    if pa.settings.use_units:
        R200c = snap.Cat.Group['Group_R_Crit200'][0].value
    else:
        R200c = snap.Cat.Group['Group_R_Crit200'][0]
    # widths = [10000, 10000, 2*R200c]
    widths = [10000, 10000, 10000]
    width_vec = (
        [2*R200c, 10000, 20000],
        [10000, 2*R200c, 20000],
        [10000, 20000, 2*R200c],
        )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3)
    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        projector = pa.Projector(snap, center, widths, direction, npix=512)

        image_file = pa.ArepoImage(projector, basedir=root_dir+'/data/',
                                   basename='projection_{}'.format(direction))

        Masses = projector.project_variable('0_Masses')
        print(Masses[0, 0])
        Volume = projector.project_variable('0_Volume')

        image_file.save_image('0_Masses', Masses)
        image_file.save_image('0_Volume', Volume)

        # snap.get_temperatures()
        TemperaturesTimesMasses = projector.project_variable(
                                snap['0_Temperatures'] * snap['0_Masses'])
        image_file.save_image('0_TemperaturesTimesMasses', TemperaturesTimesMasses)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Make a plot
        axes[ii].imshow(np.array((Masses/Volume)), origin='lower',
                        extent=np.array(projector.extent), norm=LogNorm())
    plt.show()
