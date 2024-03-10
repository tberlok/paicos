import paicos as pa
import numpy as np
from time import perf_counter
timing_gpu = np.empty(20)


def print_timing(timing_gpu):
    timing_gpu *= 1e6  # convert to us
    print(f"Elapsed time GPU: {timing_gpu.mean():.0f} ± {timing_gpu.std():.0f} us")
    timing_gpu /= 1e3  # convert to ms
    print(f"Elapsed time GPU: {timing_gpu.mean():.1f} ± {timing_gpu.std():.2f} ms")


try:
    snap = pa.Snapshot(pa.data_dir + 'highres', 247)
except FileNotFoundError as e:
    print(e)
    err_msg = ('This example requires a large data set.\nPlease see: '
               + 'https://github.com/tberlok/paicos/tree/main/data/highres/README.md'
               + ' for download instructions')
    raise RuntimeError(err_msg)
#     pr


center = np.array([499798.09375, 499879.25, 499710.65625]) * snap.length

widths = [20000, 20000, 20000]

nx = 4096

orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
projector = pa.GpuSphProjector(snap, center, widths, orientation, npix=nx, threadsperblock=64, do_pre_selection=True)

print(f'Particles to be projected: {projector.pos.shape[0]}')
print(f'Image pixels: {projector.npix_width, projector.npix_height}')

# Timing
for i in range(timing_gpu.size):
    tic = perf_counter()
    mass_image = projector.project_variable('0_Masses')
    toc = perf_counter()
    timing_gpu[i] = toc - tic
print_timing(timing_gpu)

if True:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    dens = projector.project_variable('0_Masses') \
        / projector.project_variable('0_Volume')
    extent = projector.centered_extent
    extent = extent.to('Mpc').to_physical

    plt.figure(1)
    plt.clf()
    plt.imshow(dens.value, extent=extent.value, norm=LogNorm())
    plt.xlabel(extent.label())
    plt.ylabel(extent.label())

    plt.savefig('gpu_dens.png', dpi=1400)
