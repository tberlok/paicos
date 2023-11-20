import paicos as pa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ffmpeg -framerate 10 -i frame_%d.png -vf scale=-2:720 -pix_fmt yuv420p movie.mp4 -y
# ffmpeg -framerate 10 -i frame_rotate_y_%d.png -vf scale=-2:720 -pix_fmt yuv420p movie_rotate_around_y.mp4 -y
# ffmpeg -framerate 10 -i frame_rotate_z_%d.png -vf scale=-2:720 -pix_fmt yuv420p movie_rotate_around_z.mp4 -y

plt.rc('image', origin='lower', interpolation='None')
plt.rc('text', usetex=True)

snap = pa.Snapshot(pa.root_dir + 'data/snap_247.hdf5')
center = snap.Cat.Group['GroupPos'][0]
R200c = snap.Cat.Group['Group_R_Crit200'][0].value
widths = [15000, 15000, 15000]

step_degrees = 1.8
for frame in range(201):
    orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
    angle = step_degrees * frame
    orientation.rotate_around_z(degrees=angle)
    projector = pa.NestedProjector(snap, center, widths, orientation, npix=512, make_snap_with_selection=False)
    extent = projector.extent.to('Mpc')
    rho = projector.project_variable('0_Masses') / projector.project_variable('0_Volume')
    rho = rho.to_physical.to('g cm^-3')

    if frame == 0:
        vmax = rho.max().value
        vmin = vmax * 1e-4
        plt.figure(num=1)
        plt.clf()
        fig, axes = plt.subplots(num=1)
        ima = axes.imshow(rho.value, cmap='YlGnBu', extent=extent.value, norm=LogNorm()) #vmin=vmin, vmax=vmax))
        plt.xlabel(extent.label(r'\mathrm{Perp vector1}'))
        plt.ylabel(extent.label(r'\mathrm{Perp vector2}'))
        cbar = fig.colorbar(ima, fraction=0.025, pad=0.04, label=rho.label(r'\rho'))
    else:
        ima.set_data(rho.value)
    axes.set_title(r"Rotate around $z$-axis, $\theta =$" + f"{angle:1.1f}" + r"$^\circ$")
    fig.savefig(f'frame_rotate_z_{frame}.png', dpi=700, bbox_inches='tight')
    # plt.show()
