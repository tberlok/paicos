import pytest

cupy = pytest.importorskip("cupy")
numba = pytest.importorskip("numba")


def test_gpu_sph_projector(show=False):
    """
    We compare the CPU and GPU implementations of SPH-projection.
    """
    import paicos as pa
    import numpy as np
    pa.use_units(True)

    # Load snapshot
    snap = pa.Snapshot(pa.data_dir, 247)

    center = snap.Cat.Group['GroupPos'][0]
    R200c = snap.Cat.Group['Group_R_Crit200'][0]
    widths = np.array([10000, 10000, 10000]) * R200c.uq

    # Pixels along horizontal direction
    npix = 256

    # Do some arbitrary orientation
    orientation = pa.Orientation(
        normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
    orientation.rotate_around_normal_vector(degrees=20)
    orientation.rotate_around_perp_vector1(degrees=35)
    orientation.rotate_around_perp_vector2(degrees=-47)

    # Initialize projectors
    projector = pa.NestedProjector(snap, center, widths, orientation,
                                   npix=npix, factor=2, npix_min=npix // 16,
                                   store_subimages=True)
    gpu_projector = pa.GpuSphProjector(snap, center, widths, orientation,
                                       npix=npix, threadsperblock=8,
                                       do_pre_selection=False)

    # Project density
    dens = projector.project_variable(
        '0_Masses') / projector.project_variable('0_Volume')
    gpu_dens = gpu_projector.project_variable(
        '0_Masses') / gpu_projector.project_variable('0_Volume')

    max_rel_err = np.max(np.abs(dens.value - gpu_dens.value) / dens.value)

    # Four percent tolerance is not exactly amazing,
    # but we do have different algorithms...
    assert max_rel_err < 0.04, max_rel_err

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3, sharex=True, sharey=True)
        axes[0].imshow(
            dens.value, extent=projector.centered_extent.value, norm=LogNorm())
        axes[1].imshow(
            gpu_dens.value, extent=projector.centered_extent.value, norm=LogNorm())
        axes[2].imshow(np.abs(dens.value - gpu_dens.value) / dens.value,
                       extent=projector.centered_extent.value)  # , norm=LogNorm())
        plt.show()


if __name__ == '__main__':
    test_gpu_sph_projector(True)
