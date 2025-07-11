
# def test_tree_projector_vs_ray_projector(show=False):
show = True
import paicos as pa
import numpy as np

snap = pa.Snapshot(pa.data_dir, 7,
                   basename='small_non_comoving')

widths = snap.box_size.copy
center = snap.box_size.copy / 2

tree_projector = pa.TreeProjector(snap, center, widths, 'z', npix=300)
projector = pa.RayProjector(snap, center, widths, 'z', npix=300, tol=2)

# tree_M = tree_projector.project_variable('0_Masses')
# tree_V = tree_projector.project_variable('0_Volume')
# tree_rho = tree_M / tree_V
tree_rho = tree_projector.project_variable('0_Density', additive=False)

rho = projector.project_variable('0_Density', additive=False)
# V = projector.project_variable('0_Volume', additive=False)
# rho = M / V

rel_rho = (np.abs(tree_rho - rho) / tree_rho)
# rel_M = (np.abs(tree_M - M) / tree_M)

# assert np.max(rel_rho.value) < 0.17, np.max(rel_rho.value)

# Only check away from boundaries since we do not do these correctly
# with the SPH projector.
# M_err = np.max(rel_M[35:65, 100:200])
# assert M_err < 0.16, M_err

if show:
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, nrows=3, sharex=True)

    # Define new imshow_wo_units that automatically plots the values
    fig.suptitle('Density plots')
    imshow_wo_units = pa.util.remove_astro_units(axes[0].imshow)
    imshow_wo_units(rho, extent=projector.extent,
                    vmin=rho.min(), vmax=rho.max())
    axes[0].set_ylabel(projector.extent.label('y'))
    axes[0].set_title('RayProjector')

    imshow_wo_units = pa.util.remove_astro_units(axes[1].imshow)
    imshow_wo_units(tree_rho.value, extent=tree_projector.extent.value,
                    vmin=tree_rho.min(), vmax=tree_rho.max())
    axes[1].set_ylabel(tree_projector.extent.label('y'))
    axes[1].set_title('TreeProjector')

    axes[2].imshow(rel_rho.value, extent=tree_projector.extent.value)
    axes[2].set_ylabel(tree_projector.extent.label('y'))
    axes[2].set_xlabel(tree_projector.extent.label('x'))
    axes[2].set_title('Rel difference')

    plt.show()


# if __name__ == '__main__':
#     test_tree_projector_vs_ray_projector(True)
