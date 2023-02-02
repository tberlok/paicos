

def test_projector(show=False):
    import paicos as pa
    import numpy as np

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3)

    for use_units in [False, True]:

        pa.use_units(use_units)

        basedir = pa.root_dir + '/data/'
        snap = pa.Snapshot(basedir, 247, basename='reduced_snap',
                           load_catalog=False)
        center = [398968.4, 211682.6, 629969.9]
        widths = [2000, 2000, 2000]

        for ii, direction in enumerate(['x', 'y', 'z']):

            projector = pa.Projector(snap, center, widths, direction, npix=512)

            basename = 'reduced_projection_{}'.format(direction)
            image_file = pa.ArepoImage(projector, basedir=basedir,
                                       basename=basename)

            Masses = projector.project_variable('0_Masses')

            Volume = projector.project_variable('0_Volume')

            image_file.save_image('0_Masses', Masses)
            image_file.save_image('0_Volume', Volume)

            # Move from temporary filename to final filename
            image_file.finalize()

            # Make a plot
            if show:
                axes[ii].imshow(np.array((Masses / Volume)), origin='lower',
                                extent=np.array(projector.extent),
                                norm=LogNorm())

        if show:
            plt.show()


if __name__ == '__main__':
    test_projector(True)
