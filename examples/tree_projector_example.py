import paicos as pa
import numpy as np
import matplotlib.pyplot as plt

snap = pa.Snapshot(pa.data_dir, 7,
                   basename='small_non_comoving')

widths = snap.box_size.copy
center = snap.box_size.copy / 2

tree_projector = pa.TreeProjector(snap, center, widths, 'z', npix=300)
projector = pa.Projector(snap, center, widths, 'z', npix=300, nvol=24)


tree_M = tree_projector.project_variable('0_Masses')
tree_V = tree_projector.project_variable('0_Volume')
tree_rho = tree_M / tree_V
#    rho = tree_projector.project_variable('0_Density', extrinsic=False)

M = projector.project_variable('0_Masses')
V = projector.project_variable('0_Volume')
rho = M / V

rel_rho = (np.abs(tree_rho - rho) / tree_rho)
rel_M = (np.abs(tree_M - M) / tree_M)

for ii in range(1, 3):

    plt.figure(ii)
    plt.clf()
    fig, axes = plt.subplots(num=ii, nrows=3, sharex=True)

    if ii == 1:
        # Define new imshow_wo_units that automatically plots the values
        fig.suptitle('Density plots (by dividing additive projections)')
        imshow_wo_units = pa.util.remove_astro_units(axes[0].imshow)
        imshow_wo_units(rho, extent=projector.extent,
                        vmin=rho.min(), vmax=rho.max())
        axes[0].set_ylabel(projector.extent.label('y'))
        axes[0].set_title('Sph projection')

        imshow_wo_units = pa.util.remove_astro_units(axes[1].imshow)
        imshow_wo_units(tree_rho.value, extent=tree_projector.extent.value,
                        vmin=rho.min(), vmax=rho.max())
        axes[1].set_ylabel(tree_projector.extent.label('y'))
        axes[1].set_title('Tree projection')

        axes[2].imshow(rel_rho.value, extent=tree_projector.extent.value)
        axes[2].set_ylabel(tree_projector.extent.label('y'))
        axes[2].set_xlabel(tree_projector.extent.label('x'))
        axes[2].set_title('Rel difference')

    if ii == 2:
        fig.suptitle('Mass plots')
        # Define new imshow_wo_units that automatically plots the values
        imshow_wo_units = pa.util.remove_astro_units(axes[0].imshow)
        imshow_wo_units(M, extent=projector.extent)
        axes[0].set_ylabel(projector.extent.label('y'))
        axes[0].set_title('Sph projection')

        imshow_wo_units = pa.util.remove_astro_units(axes[1].imshow)
        imshow_wo_units(tree_M.value, extent=tree_projector.extent.value)
        axes[1].set_ylabel(tree_projector.extent.label('y'))
        axes[1].set_title('Tree projection')

        axes[2].imshow(rel_M.value, extent=tree_projector.extent.value)
        axes[2].set_ylabel(tree_projector.extent.label('y'))
        axes[2].set_xlabel(tree_projector.extent.label('x'))
        axes[2].set_title('Rel difference')

plt.show()
