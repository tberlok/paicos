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
        [2 * R200c, 10000, 20000],
        [10000, 2 * R200c, 20000],
        [10000, 20000, 2 * R200c],
    )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=3)
    for ii, direction in enumerate(['x', 'y', 'z']):
        print(ii)
        widths = width_vec[ii]

        projector = pa.Projector(snap, center, widths, direction, npix=512,
                                 parttype=1, make_snap_with_selection=True)
        treeprojector = pa.TreeProjector(snap, center, widths, direction,
                                         npix=512, parttype=1, verbose=True)

        image_file = pa.ArepoImage(projector, basedir=root_dir + 'test_data/',
                                   basename=f'projection_{direction}')

        Density = projector.project_variable('1_SubfindDMDensity')
        Density_tree = treeprojector.project_variable('1_SubfindDMDensity',
                                                      extrinsic=False)

        image_file.save_image('1_SubfindDMDensity', Density)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Make a plot
        axes[0,ii].imshow(np.array(Density), origin='lower',
                          extent=np.array(projector.extent), norm=LogNorm())
        axes[0, 1].set_title('SPH projector')
        axes[1,ii].imshow(np.array(Density_tree), origin='lower',
                          extent=np.array(projector.extent), norm=LogNorm())
        axes[1, 1].set_title('Tree projector')

    plt.show()
