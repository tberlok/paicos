import numpy as np
from . import util

"""
Files to modify to implement these changes:
arepo_image
image_creators
all projectors
the slicer
arepo snapshot

TODO: Make it possible to pass angles, so that one may easily
change the viewing inclination in small steps in angle.
"""


@util.remove_astro_units
def normalize_vector(vector):
    """
    Normalize a vector and return it as a numpy array
    """
    return vector / np.linalg.norm(vector)


@util.remove_astro_units
def find_a_perpendicular_vector(vector):
    """
    Given an input vector, find a vector which is perpendicular.
    Assumes 3D vectors.
    """
    assert np.linalg.norm(vector) > 0.
    if vector[1] == 0 and vector[2] == 0:
        return np.cross(vector, [0, 1, 0])
    return np.cross(vector, [1, 0, 0])


def _assert_perpendicular(vector1, vector2, tol=1e-15):
    assert abs(np.dot(vector1, vector2)) < tol


@util.remove_astro_units
def get_basis_from_2vecs(vector1, vector2):
    """
    Returns a right-hand orthogonal basis, e1, e2, e3,
    where e1 is in the direction of vector1.
    Adapted from code by Ewald Puchwein.
    """

    # unit vector in vector1 direction
    e1 = vector1 / np.linalg.norm(vector1)

    e2 = vector2 - np.dot(e1, vector2) * e1
    # unit vector along direction of vector2-component perpendicular to vector1
    e2 = e2 / np.linalg.norm(e2)

    e3 = np.cross(e1, e2)
    assert np.fabs(np.linalg.norm(e3) - 1.0) < 1.0e-10

    # 3rd otrhogonal unit vector
    e3 = e3 / np.linalg.norm(e3)

    _assert_perpendicular(e1, e2)
    _assert_perpendicular(e2, e3)
    _assert_perpendicular(e3, e1)

    return e1, e2, e3


class Orientation:
    def __init__(self, normal_vector=None, perp_vector1=None):
        """
        Inputs:

        normal_vector: a vector which will be normal to the face
                       of the image, e.g. pass the angular momentum vector
                       of a galaxy here for a face on projection.

        perp_vector1: Perpendicular vector to the image plane, e.g.,
                         pass the angular momentum vector here for an edge-on
                         projection.

        It's possible to supply both inputs.
        """

        # Construct unit vectors for the chosen coordinate system
        self._construct_unit_vectors(normal_vector, perp_vector1)

    def _construct_unit_vectors(self, normal_vector, perp_vector1):
        """
        Construct right-handed coordinate system
        """

        if normal_vector is None and perp_vector1 is None:
            err_msg = "You have to set either normal_vector, perp_vector1 or both"
            raise RuntimeError(err_msg)

        elif normal_vector is not None and perp_vector1 is not None:
            pass
        else:
            # Normal vector given
            if normal_vector is not None:
                perp_vector1 = find_a_perpendicular_vector(normal_vector)
            # in plane vector given
            elif perp_vector1 is not None:
                normal_vector = find_a_perpendicular_vector(perp_vector1)

        # Get basis vectors
        e1, e2, e3 = get_basis_from_2vecs(normal_vector, perp_vector1)

        # Set unitless unit-vectors
        self.normal_vector = e1
        self.perp_vector1 = e2
        self.perp_vector2 = e3

    def rotate_orientation_around_x(degrees=None, radians=None):
        """
        This should update all the unit vectors and the rotation matrix.
        Since these are properties, I guess updating 
        self.perp_vector2
        self.perp_vector1
        self.normal_vector
        should be enough?
        """
        pass

    def rotate_orientation_around_y(degrees=None, radians=None):
        pass

    def rotate_orientation_around_z(degrees=None, radians=None):
        pass

    @property
    def inverse_rotation_matrix(self):
        """
        An inverse rotation matrix, which rotates from the Orientation
        system into a standard xyz coordinate system. We have the
        following properties:

        np.matmul(inverse_rotation_matrix, perp_vector1) = [1, 0, 0]
        np.matmul(inverse_rotation_matrix, perp_vector2) = [0, 1, 0]
        np.matmul(inverse_rotation_matrix, normal_vector) = [0, 0, 1]

        The use case for the inverse rotation matrix is to transform
        Arepo data into a coordinate system where the perp_vector1 direction
        is in the x-direction, the perp_vector2 direction is in the y-direction
        and the normal_vector is in the z-direction.
        """
        ex = self.cartesian_unit_vectors['x']
        ey = self.cartesian_unit_vectors['y']
        ez = self.cartesian_unit_vectors['z']
        return np.array([ex, ey, ez])

    @property
    def rotation_matrix(self):
        """
        This rotation matrix rotates the standard cartesian unit vectors

        ex = [1, 0, 0]
        ey = [0, 1, 0]
        ez = [0, 0, 1]

        into the cartesian coordinate system defined by the three unit vectors

        perp_vector1
        perp_vector2
        normal_vector

        which can also be accessed through the cartesian_unit_vectors method.

        That is, the following equality holds:

        perp_vector1 = np.matmul(rotation_matrix, [1, 0, 0])
        perp_vector2 = np.matmul(rotation_matrix, [0, 1, 0])
        normal_vector = np.matmul(rotation_matrix, [0, 0, 1])
        """
        return self.inverse_rotation_matrix.T

    @property
    def cartesian_unit_vectors(self):
        """
        The x, y, and z unit vectors in the Orientation
        given in terms of their components in the original simulation
        coordinate system.
        """
        unit_vectors = {}
        unit_vectors['x'] = self.perp_vector1
        unit_vectors['y'] = self.perp_vector2
        unit_vectors['z'] = self.normal_vector
        return unit_vectors

    @property
    def spherical_unit_vectors(self):
        raise RuntimeError('not implemented')

    @property
    def cylindrical_unit_vectors(self):
        raise RuntimeError('not implemented')

    @property
    def euler_angles(self):
        # TODO: intrinsic and extrinsic version?
        raise RuntimeError('not implemented')
