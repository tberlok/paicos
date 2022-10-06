import numpy as np


class Projector:
    """
    This implements an SPH-like projection of gas variables.
    """

    def __init__(self, arepo_snap, center, widths, direction,
                 npix=512, nvol=8):
        from cython_functions import get_index_of_region
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

    def project_variable(self, func):
        from cython_functions import project_image

        if type(func) is str:
            if func == 'Masses':
                self.snap.load_data(0, 'Masses')
                variable = self.snap.P['0_Masses']
            elif func == 'GFM_MetallicityTimesMasses':
                self.snap.load_data(0, 'GFM_Metallicity')
                variable = self.snap.P['0_Masses']*self.snap.P['0_GFM_Metallicity']
            elif func == 'Volume':
                self.snap.load_data(0, 'Masses')
                self.snap.load_data(0, 'Density')
                variable = self.snap.P['0_Masses']/self.snap.P['0_Density']
            elif func == 'EnergyDissipation':
                self.snap.load_data(0, 'EnergyDissipation')
                variable = self.snap.P['0_EnergyDissipation']
            elif func == 'MachnumberTimesEnergyDissipation':
                self.snap.load_data(0, 'Machnumber')
                self.snap.load_data(0, 'EnergyDissipation')
                variable = self.snap.P['0_Machnumber']*self.snap.P['0_EnergyDissipation']
            elif func == 'MagneticFieldSquaredTimesVolume':
                self.snap.load_data(0, 'MagneticField')
                variable = self.snap.P["0_Volumes"]*np.sum(self.snap.P['0_MagneticField']**2, axis=1)
            elif func == 'PressureTimesVolume':
                self.snap.load_data(0, 'InternalEnergy')
                self.snap.load_data(0, 'Density')
                gamma = 5/3
                # thermal pressure times volume
                # variable = self.snap.P["0_Volumes"] * self.snap.P["0_InternalEnergy"] * self.snap.P["0_Density"] * (gamma - 1.)
                # Same as above but faster
                variable = self.snap.P["0_Masses"] * self.snap.P["0_InternalEnergy"] * (gamma - 1.)

            elif func == 'TemperatureTimesMasses':
                self.snap.get_temperatures()
                variable = self.snap.P["0_Temperatures"]*self.snap.P['0_Masses']

            elif func == 'EnstrophyTimesMasses':
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
                raise RuntimeError('unknown function requested', func)

        else:
            variable = func(self.arepo_snap)

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
                                       boxsize)
        elif self.direction == 'y':
            projection = project_image(self.pos[self.index, 0],
                                       self.pos[self.index, 2],
                                       variable[self.index],
                                       self.hsml, self.npix,
                                       xc, zc, self.width_x, self.width_z,
                                       boxsize)
        elif self.direction == 'z':
            projection = project_image(self.pos[self.index, 0],
                                       self.pos[self.index, 1],
                                       variable[self.index],
                                       self.hsml, self.npix,
                                       xc, yc, self.width_x, self.width_y,
                                       boxsize)
        # image = projection()
        return projection

    # def sph_like_projection(arepo_snap, variable_func, center, widths, dir):
    #     pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import load_partial_snap
    from arepo_image import ArepoImage

    snap = load_partial_snap.snapshot('.', 247)
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
