import pytest


@pytest.mark.order(1)
def test_slicer_writer_and_reader(show=False):

    import paicos as pa

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

    snap = pa.Snapshot(pa.root_dir + '/data', 247,
                       basename='reduced_snap', load_catalog=False)
    center = [398968.4, 211682.6, 629969.9]

    width_vec = (
        [0.0, 2000, 2000],
        [2000, 0.0, 2000],
        [2000, 2000, 0.0],
    )

    if show:
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3)

    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        slicer = pa.Slicer(snap, center, widths, direction, npix=512)

        image_file = pa.ArepoImage(slicer, basedir=pa.root_dir + 'test_data/',
                                   basename=f'red_slice_{direction}')

        Density = slicer.slice_variable(snap['0_Density'])
        Volume = slicer.slice_variable('0_Volume')

        image_file.save_image('0_Density', Density)

        # Move from temporary filename to final filename
        image_file.finalize()

        # Create a new image object in amend mode
        image_file = pa.ArepoImage(slicer, basedir=pa.root_dir + 'test_data/',
                                   basename=f'red_slice_{direction}',
                                   mode='a')

        # Now add the temperatures as well
        image_file.save_image('0_Volume', Volume)

        if show:
            # Make a plot
            if pa.settings.use_units:
                axes[ii].imshow(Density.value, origin='lower',
                                extent=slicer.extent.value, norm=LogNorm())
            else:
                axes[ii].imshow(Density, origin='lower',
                                extent=slicer.extent, norm=LogNorm())

    # Example of how to read the image files
    im = pa.ImageReader(pa.root_dir + 'test_data', 247, 'red_slice_x')

    im['0_Volume']

    if show:
        plt.show()


if __name__ == '__main__':
    test_slicer_writer_and_reader(True)
