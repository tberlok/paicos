
def test_arepo_converter():
    import paicos as pa
    import numpy as np
    converter = pa.ArepoConverter(pa.root_dir + '/data/slice_x_247.hdf5')

    rho = np.array([2, 4])
    rho = converter.get_paicos_quantity(rho, 'Density')

    Mstars = 1
    Mstars = converter.get_paicos_quantity(Mstars, 'Masses')

    B = converter.get_paicos_quantity([1], 'MagneticField')

    T = converter.get_paicos_quantity([1], 'Temperature')

    B = converter.get_paicos_quantity([1], 'MagneticField')

    v = converter.get_paicos_quantity([2], 'Velocities')

    np.testing.assert_allclose(T.to('keV').value, np.array([8.61733326e-08]))
    np.testing.assert_allclose(B.to('uG').value, np.array([2.6019053]))
