import pytest

cupy = pytest.importorskip("cupy")
numba = pytest.importorskip("numba")


def test_gpu_ray_projector(show=False):
    """
    We compare the CPU and GPU implementations of ray-tracing.
    """
    import paicos as pa
    import numpy as np
    pa.use_units(True)

    # Load snapshot
    snap = pa.Snapshot(pa.data_dir, 247)

    center = snap.Cat.Group['GroupPos'][0]
    R200c = snap.Cat.Group['Group_R_Crit200'][0]
    widths = np.array([4000, 4000, 4000]) * R200c.uq

    # Pixels along horizontal direction
    npix = 256

    # Do some arbitrary orientation
    orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
    orientation.rotate_around_normal_vector(degrees=20)
    orientation.rotate_around_perp_vector1(degrees=35)
    orientation.rotate_around_perp_vector2(degrees=-47)

    # Initialize projectors
    tree_projector = pa.TreeProjector(snap, center, widths, orientation, npix=npix,
                                      tol=0.25)
    gpu_projector = pa.GpuRayProjector(snap, center, widths, orientation, npix=npix,
                                       threadsperblock=8, do_pre_selection=True,
                                       tol=0.25)

    # Project density
    tree_dens = tree_projector.project_variable('0_Density', additive=False)
    gpu_dens = gpu_projector.project_variable('0_Density', additive=False)

    max_rel_err = np.max(np.abs(tree_dens.value - gpu_dens.value) / tree_dens.value)

    # Seven percent tolerance is not exactly amazing,
    # but we do have different algorithms...
    assert max_rel_err < 0.07, max_rel_err

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3, sharex=True, sharey=True)
        axes[0].imshow(tree_dens.value, extent=tree_projector.centered_extent.value,
                       norm=LogNorm())
        axes[1].imshow(gpu_dens.value, extent=tree_projector.centered_extent.value,
                       norm=LogNorm())
        axes[2].imshow(np.abs(tree_dens.value - gpu_dens.value) / tree_dens.value,
                       extent=tree_projector.centered_extent.value)
        # plt.savefig('gpu_ray_tracer_test.png')
        plt.show()


if __name__ == '__main__':
    test_gpu_ray_projector(True)
