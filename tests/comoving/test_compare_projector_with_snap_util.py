

def test_compare_projector_with_snap_util(show=False):
    """
    We compare the Paicos tree projector with the arepo-snap-util software
    commonly used for arepo analysis. We load a data file which was generated with
    the following code:

    import gadget
    import numpy as np

    g = gadget.gadget_readsnap(247, snappath=pa.root_dir + 'data',
                               snapbase='snap_')

    center = np.array([398968.4, 211682.6, 629969.9])

    box = np.array([2000, 2000, 2000])

    sl = g.get_Aslice('rho', res=128, numthreads=1, center=center / g.hubbleparam,
                  box=box / g.hubbleparam, proj=True)

    np.savez(pa.root_dir + 'data/arepo-snap-util-slice-proj.npz', sl['grid'].T)
    """
    import numpy as np
    import paicos as pa
    snap = pa.Snapshot(pa.root_dir + 'data', 247, basename='reduced_snap')
    center = [398968.4, 211682.6, 629969.9]

    widths = np.array([2000, 2000, 2000])

    projector = pa.TreeProjector(snap, center, widths, 'z', npix=128)

    dat = np.load(pa.root_dir + '/data/arepo-snap-util-proj.npz')
    pai = projector.project_variable('0_Density', extrinsic=False).to_physical.value

    # Note division by 128 because the Paicos tree projector has different behavior
    # than the arepo-snap-util projector. TODO: Make a better test.
    are = dat['arr_0'] / 128

    rel_diff = np.abs(pai - are) / are

    rms_diff = (np.sqrt(np.mean(rel_diff)))

    assert rms_diff < 0.1, f'rms_diff {rms_diff} is too large'

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3, sharey=True)
        axes[0].imshow(pai, norm=LogNorm(), origin='lower')
        axes[1].imshow(are, norm=LogNorm(), origin='lower')

        axes[2].imshow(rel_diff, origin='lower')

        axes[0].set_title('Paicos')
        axes[1].set_title('arepo-snap-util')
        axes[2].set_title('abs. rel. difference')
        plt.show()


if __name__ == '__main__':
    test_compare_projector_with_snap_util(True)
