import numpy as np


class Projector:
    """
    This implements an SPH-like projection of gas variables.
    """

    def __init__(self, arepo_snap, center, widths, direction,
                 npix=512, nvol=8, numthreads=16):
        from paicos import get_index_of_region
        from paicos import simple_reduction

        n = simple_reduction(1000, numthreads)
        if n == 1000:
            self.use_omp = True
            self.numthreads = numthreads
        else:
            self.use_omp = False
            self.numthreads = 1
            print('OpenMP is not working on your system...')

        self.snap = arepo_snap

        self.center = center
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]

        self.widths = widths
        self.width_x = widths[0]
        self.width_y = widths[1]
        self.width_z = widths[2]

        self.direction = direction

        self.npix = npix

        snap = arepo_snap
        xc = self.xc
        yc = self.yc
        zc = self.zc
        width_x = self.width_x
        width_y = self.width_y
        width_z = self.width_z

        snap.get_volumes()
        snap.load_data(0, "Coordinates")
        self.pos = pos = np.array(snap.P["0_Coordinates"], dtype=np.float32)

        if direction == 'x':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.yc - self.width_y/2, self.yc + self.width_y/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'y':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.zc - self.width_z/2, self.zc + self.width_z/2]

        elif direction == 'z':
            self.index = get_index_of_region(pos, xc, yc, zc,
                                             width_x, width_y, width_z,
                                             snap.box)
            self.extent = [self.xc - self.width_x/2, self.xc + self.width_x/2,
                           self.yc - self.width_y/2, self.yc + self.width_y/2]

        self.hsml = np.cbrt(nvol*(snap.P["0_Volumes"][self.index]) /
                            (4.0*np.pi/3.0))

    def _get_variable(self, variable_str):

        if type(variable_str) is str:
            if variable_str == 'Masses':
                self.snap.load_data(0, 'Masses')
                variable = self.snap.P['0_Masses']
            elif variable_str == 'GFM_MetallicityTimesMasses':
                self.snap.load_data(0, 'GFM_Metallicity')
                variable = self.snap.P['0_Masses']*self.snap.P['0_GFM_Metallicity']
            elif variable_str == 'Volume':
                self.snap.load_data(0, 'Masses')
                self.snap.load_data(0, 'Density')
                variable = self.snap.P['0_Masses']/self.snap.P['0_Density']
            elif variable_str == 'EnergyDissipation':
                self.snap.load_data(0, 'EnergyDissipation')
                variable = self.snap.P['0_EnergyDissipation']
            elif variable_str == 'MachnumberTimesEnergyDissipation':
                self.snap.load_data(0, 'Machnumber')
                self.snap.load_data(0, 'EnergyDissipation')
                variable = self.snap.P['0_Machnumber']*self.snap.P['0_EnergyDissipation']
            elif variable_str == 'MagneticFieldSquaredTimesVolume':
                self.snap.load_data(0, 'MagneticField')
                variable = self.snap.P["0_Volumes"]*np.sum(self.snap.P['0_MagneticField']**2, axis=1)
            elif variable_str == 'PressureTimesVolume':
                self.snap.load_data(0, 'InternalEnergy')
                self.snap.load_data(0, 'Density')
                gamma = 5/3
                # thermal pressure times volume
                # variable = self.snap.P["0_Volumes"] * self.snap.P["0_InternalEnergy"] * self.snap.P["0_Density"] * (gamma - 1.)
                # Same as above but faster
                variable = self.snap.P["0_Masses"] * self.snap.P["0_InternalEnergy"] * (gamma - 1.)

            elif variable_str == 'TemperatureTimesMasses':
                self.snap.get_temperatures()
                variable = self.snap.P["0_Temperatures"]*self.snap.P['0_Masses']

            elif variable_str == 'EnstrophyTimesMasses':
                # absolute vorticity squared times one half ("enstrophy")
                self.snap.load_data(0, 'VelocityGradient')

                n_cells = self.snap.P['0_VelocityGradient'].shape[0]
                # Reshape to tensor form
                gradV = self.snap.P['0_VelocityGradient'][()].reshape(n_cells, 3, 3)
                # Get vorticity components
                vor_x = gradV[:, 2, 1] - gradV[:, 1, 2]
                vor_y = gradV[:, 0, 2] - gradV[:, 2, 0]
                vor_z = gradV[:, 1, 0] - gradV[:, 0, 1]

                # The vorticity vector
                # vorticity = np.stack([vor_x, vor_y, vor_z], axis=1)

                enstrophy = 0.5 * (vor_x**2 + vor_y**2 + vor_z**2)
                variable = enstrophy*self.snap.P['0_Masses']
        else:
            raise RuntimeError('unknown function requested', variable_str)

        return variable

    def project_variable(self, variable):

        if self.use_omp:
            from paicos import project_image_omp as project_image
        else:
            from paicos import project_image

        import types
        if isinstance(variable, str):
            variable = self._get_variable(variable)
        elif isinstance(variable, types.FunctionType):
            variable = variable(self.arepo_snap)
        elif isinstance(variable, np.ndarray):
            pass
        else:
            raise RuntimeError('Unexpected type for variable')

        xc = self.xc
        yc = self.yc
        zc = self.zc
        boxsize = self.snap.box
        if self.direction == 'x':
            projection = project_image(self.pos[self.index, 1],
                                       self.pos[self.index, 2],
                                       variable[self.index],
                                       self.hsml, self.npix,
                                       yc, zc, self.width_y, self.width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'y':
            projection = project_image(self.pos[self.index, 0],
                                       self.pos[self.index, 2],
                                       variable[self.index],
                                       self.hsml, self.npix,
                                       xc, zc, self.width_x, self.width_z,
                                       boxsize, self.numthreads)
        elif self.direction == 'z':
            projection = project_image(self.pos[self.index, 0],
                                       self.pos[self.index, 1],
                                       variable[self.index],
                                       self.hsml, self.npix,
                                       xc, yc, self.width_x, self.width_y,
                                       boxsize, self.numthreads)
        return projection


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from paicos import load_partial_snap
    from paicos import ArepoImage

    snap = load_partial_snap.snapshot('../data', 247)
    center = snap.Cat.Group['GroupPos'][0]
    R200c = snap.Cat.Group['Group_R_Crit200'][0]
    # widths = [10000, 10000, 2*R200c]
    widths = [10000, 10000, 10000]
    width_vec = (
        [2*R200c, 10000, 10000],
        [10000, 2*R200c, 10000],
        [10000, 10000, 2*R200c],
        )

    # width_vec = (
    #     [2*R200c, 2*R200c, 2*R200c],
    #     [2*R200c, 2*R200c, 2*R200c],
    #     [2*R200c, 2*R200c, 2*R200c],
    #     )

    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=3)#, sharey=True)
    for ii, direction in enumerate(['x', 'y', 'z']):
        widths = width_vec[ii]
        p = Projector(snap, center, widths, direction, npix=512)

    # widths = [10000, 2*R200c, 10000]
    # p = Projector(snap, center, widths, 'y', npix=512)
        image_file = ArepoImage('projection_{}.hdf5'.format(direction),
                                p.snap.filename, center, widths, direction)

        Masses = p.project_variable('Masses')
        Volume = p.project_variable('Volume')

        image_file.save_image('Masses', Masses)
        image_file.save_image('Volume', Volume)

        # Move from temporary filename to final filename
        image_file.finalize()
        axes[ii].imshow(np.log10(Masses/Volume).T, origin='lower',
                        extent=p.extent)
    plt.show()
