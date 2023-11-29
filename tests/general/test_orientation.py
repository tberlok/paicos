import paicos as pa
import numpy as np

"""
Some simple tests of the Orientation class
"""


def assert_equal_orientation(o1, o2):

    np.testing.assert_allclose(o1.cartesian_unit_vectors['x'],
                               o2.cartesian_unit_vectors['x'], atol=1e-15)
    np.testing.assert_allclose(o1.cartesian_unit_vectors['y'],
                               o2.cartesian_unit_vectors['y'], atol=1e-15)
    np.testing.assert_allclose(o1.cartesian_unit_vectors['z'],
                               o2.cartesian_unit_vectors['z'], atol=1e-15)


for ii in range(4):
    o1 = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
    o2 = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])

    assert_equal_orientation(o1, o2)

    if ii == 0:
        o1.rotate_around_x(23)
        o2.rotate_around_perp_vector1(23)
    elif ii == 1:
        o1.rotate_around_y(84)
        o2.rotate_around_perp_vector2(84)
    elif ii == 2:
        o1.rotate_around_z(43)
        o2.rotate_around_normal_vector(43)
    elif ii == 3:
        o1.rotate_around_y(90)
        o1.rotate_around_x(30)

        o2.rotate_around_perp_vector2(90)
        o2.rotate_around_normal_vector(30)

    else:
        raise RuntimeError('not checked')

    assert_equal_orientation(o1, o2)
    o2.rotate_around_normal_vector(30)
    assert o1._are_equal(o2) is False
    assert o1._are_equal(o1.copy)
