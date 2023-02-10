
def test_non_comoving(show=False):
    import paicos as pa
    snap = pa.Snapshot(pa.root_dir + '/data/', 7,
                       basename='small_non_comoving')

    widths = snap.box_size.copy
    center = snap.box_size.copy / 2
    print(widths)
    projector = pa.Projector(snap, center, widths, 'z', npix=300, nvol=128)
    rho_proj = projector.project_variable('0_Density')

    widths[2] = 0
    slicer = pa.Slicer(snap, center.value, widths.value, 'z')
    rho_slice = slicer.slice_variable('0_Density')

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2, sharex=True)
        axes[0].imshow(rho_slice.value, extent=slicer.extent.value)
        axes[0].set_ylabel(slicer.extent.label('y'))
        axes[0].set_title('slice')

        axes[1].imshow(rho_proj.value, extent=projector.extent.value)
        axes[1].set_ylabel(projector.extent.label('y'))
        axes[1].set_xlabel(projector.extent.label('x'))
        axes[1].set_title('projection')
        plt.show()

    snap['0_Density']
    snap['0_Masses']

    snap['0_Masses'].no_small_h

    snap['0_Density'].to_physical


if __name__ == '__main__':
    test_non_comoving(True)
