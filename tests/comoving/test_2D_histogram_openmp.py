

def test_2D_histogram_openmp(show=False):
    import numpy as np
    import paicos as pa
    pa.use_units(True)

    snap = pa.Snapshot(pa.root_dir + '/data', 247, basename='reduced_snap',
                       load_catalog=False)
    center = [398968.4, 211682.6, 629969.9] * snap.length

    r = np.sqrt(np.sum((snap['0_Coordinates'] - center[None, :])**2., axis=1))

    r = r.to_physical.astro
    rho = snap['0_Density'].to_physical.to('g cm^-3')
    M = snap['0_Masses'].to_physical

    # Create histogram object
    r_rho_vec = []

    for ii, numthreads in enumerate([1, 4]):
        pa.numthreads(numthreads)
        r_rho = pa.Histogram2D(snap, r, rho, weights=M, bins_x=100,
                               bins_y=100, logscale=True)

        # Create colorlabel
        r_rho.get_colorlabel(r'r', '\\rho', 'M')

        # Save the 2D histogram
        r_rho.save(basedir=pa.root_dir + 'test_data',
                   basename=f'r_rho_hist_{numthreads}')

        del r_rho

        r_rho_vec.append(pa.Histogram2DReader(pa.root_dir + 'test_data', 247,
                                              basename=f'r_rho_hist_{numthreads}'))
        if show:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            plt.figure(ii + 1)
            plt.clf()
            r_rho = r_rho_vec[ii]
            plt.pcolormesh(r_rho.centers_x.value, r_rho.centers_y.value,
                           r_rho.hist2d.value, norm=LogNorm())
            plt.xlabel(r_rho.centers_x.label('r'))
            plt.ylabel(r_rho.centers_y.label('\\rho'))

            plt.title(f'Histogram2D, numthreads={pa.settings.numthreads_reduction}')
            if r_rho.logscale:
                plt.xscale('log')
                plt.yscale('log')
            cbar = plt.colorbar()
            cbar.set_label(r_rho.colorlabel)
    max_val = np.max(r_rho_vec[0].hist2d.value)
    max_diff = np.max(np.abs(r_rho_vec[0].hist2d.value - r_rho_vec[1].hist2d.value))
    err_estimate = max_diff / max_val
    # print(err_estimate)
    assert err_estimate < 1e-10, f'err_estimate={err_estimate} is too high'
    if show:
        plt.figure(3)
        plt.clf()
        r_rho = r_rho_vec[ii]
        diff = np.abs(r_rho_vec[0].hist2d - r_rho_vec[1].hist2d)
        plt.pcolormesh(r_rho.centers_x.value, r_rho.centers_y.value,
                       diff.value, norm=LogNorm())
        plt.xlabel(r_rho.centers_x.label('r'))
        plt.ylabel(r_rho.centers_y.label('\\rho'))

        plt.title('Abs. difference')
        if r_rho.logscale:
            plt.xscale('log')
            plt.yscale('log')
        cbar = plt.colorbar()
        cbar.set_label(r_rho.colorlabel)
        plt.show()


if __name__ == '__main__':
    test_2D_histogram_openmp(show=True)
