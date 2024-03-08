import numpy as np
from . import util

"""
The orientation clas and its helper functions.
"""


@util.remove_astro_units
def normalize_vector(vector):
    """
    Normalize a vector and return it as a numpy array

    :meta private:
    """
    return vector / np.linalg.norm(vector)


@util.remove_astro_units
def find_a_perpendicular_vector(vector):
    """
    Given an input vector, find a vector which is perpendicular.
    Assumes 3D vectors.

    :meta private:
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

    :meta private:
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
    """
    The Orientation class is used to define an orientation of e.g.
    an image. An instance of this class can be passed to an ImageCreator.
    """
    def __init__(self, normal_vector=None, perp_vector1=None):
        """

        Parameters:

            normal_vector: a vector which will be normal to the face of the image,
                           i.e. the depth direction. You can e.g. pass the angular momentum
                           vector of a galaxy for a face on projection.

            perp_vector1: a vector which will be parallel to the image plane. This
                          will become the horizontal direction of your image. You can e.g. pass
                          the angular momentum vector here for an edge-on projection.

        It's possible to supply both inputs, which then fully determines the
        orientation (the vertical direction of the image, perp_vector2, is
        automatically calculated).
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

    def _get_radians(self, degrees=None, radians=None):
        """
        Converts from degrees to radians.
        """
        if radians is not None:
            return radians
        return 2.0 * np.pi * degrees / 360.

    def _get_rotation_matrix(self, axis, degrees=None, radians=None):
        """
        Standard implementation of rotation matrices around the
        x, y, and z coordinate axes. Rotation matrices for rotation
        around the coordinate axes in the right-handed
        coordinate system (perp_vector1, perp_vector2, normal_vector)
        is done by transforming this system to a coordinate
        system where these three axes are aligned along x, y and z,
        respectively, applying the either Rx, Ry or Rz and
        then transforming back.
        """
        if radians is None:
            assert degrees is not None
            radians = self._get_radians(degrees)

        if axis == 'x':
            R = np.array([[1, 0, 0],
                         [0, np.cos(radians), -np.sin(radians)],
                         [0, np.sin(radians), np.cos(radians)]])
            return R
        elif axis == 'y':
            R = np.array([[np.cos(radians), 0, np.sin(radians)],
                         [0, 1, 0],
                         [-np.sin(radians), 0, np.cos(radians)]])
            return R
        elif axis == 'z':
            R = np.array([[np.cos(radians), -np.sin(radians), 0],
                         [np.sin(radians), np.cos(radians), 0],
                         [0, 0, 1]])
            return R
        elif axis == 'perp_vector1':
            u = self.perp_vector1
        elif axis == 'perp_vector2':
            u = self.perp_vector2
        elif axis == 'normal_vector':
            u = self.normal_vector

        else:
            raise RuntimeError("invalid input")

        # Construct rotation matrix that rotates radians around
        # unit vector 'u'
        # see e.g. https://en.wikipedia.org/w/index.php?title=Rotation_matrix
        u_cross = np.array([[0, -u[2], u[1]],
                            [u[2], 0, -u[0]],
                            [-u[1], u[0], 0]])
        R = np.cos(radians) * np.identity(3) + np.sin(radians) * u_cross \
            + (1 - np.cos(radians)) * np.outer(u, u)

        return R

    def _apply_rotation_matrix(self, R):
        self.normal_vector = np.matmul(R, self.normal_vector)
        self.perp_vector1 = np.matmul(R, self.perp_vector1)
        self.perp_vector2 = np.matmul(R, self.perp_vector2)

    def rotate_around_x(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around the 'x'-coordinate axis
        """
        R = self._get_rotation_matrix('x', degrees, radians)
        self._apply_rotation_matrix(R)

    def rotate_around_y(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around the 'y'-coordinate axis
        """
        R = self._get_rotation_matrix('y', degrees, radians)
        self._apply_rotation_matrix(R)

    def rotate_around_z(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around the 'z'-coordinate axis
        """
        R = self._get_rotation_matrix('z', degrees, radians)
        self._apply_rotation_matrix(R)

    def rotate_around_normal_vector(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around its normal_vector.

        For a projection image, this is the depth direction.
        """
        R = self._get_rotation_matrix('normal_vector', degrees, radians)
        self._apply_rotation_matrix(R)

    def rotate_around_perp_vector1(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around its perp_vector1.

        For an image, this is the horizontal axis.
        """
        R = self._get_rotation_matrix('perp_vector1', degrees, radians)
        self._apply_rotation_matrix(R)

    def rotate_around_perp_vector2(self, degrees=None, radians=None):
        """
        Rotates the orientation instance around its perp_vector2.

        For an image, this is the vertical axis.
        """
        R = self._get_rotation_matrix('perp_vector2', degrees, radians)
        self._apply_rotation_matrix(R)

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
        This rotation matrix rotates the standard cartesian unit vectors::

            ex = [1, 0, 0]
            ey = [0, 1, 0]
            ez = [0, 0, 1]

        into the cartesian coordinate system defined by the three unit vectors::

            perp_vector1
            perp_vector2
            normal_vector

        which can also be accessed through the cartesian_unit_vectors method.

        That is, the following equality holds::

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
        """
        To be implemented.

        :meta private:
        """
        raise RuntimeError('not implemented')

    @property
    def cylindrical_unit_vectors(self):
        """
        To be implemented.

        :meta private:
        """
        raise RuntimeError('not implemented')

    @property
    def euler_angles(self):
        """
        To be implemented.

        :meta private:
        """
        raise RuntimeError('not implemented')

    @property
    def copy(self):
        """
        Returns a copy of the current Orientation instance.
        """
        return Orientation(normal_vector=self.normal_vector,
                           perp_vector1=self.perp_vector1)

    def _are_equal(self, orientation2):
        """
        Compare the current orientation instance with a
        different one. Returns True if they are identical
        and False if not.
        """
        u1 = self.cartesian_unit_vectors
        u2 = orientation2.cartesian_unit_vectors
        identical = True
        for key in u1:
            identical = np.allclose(u1[key], u2[key], atol=1e-15) and identical
        return identical

    def __print__(self):
        """
        Useful for displaying an Orientation instance.

        :meta private:
        """
        print('normal_vector:', self.normal_vector)
        print('perp_vector1: ', self.perp_vector1)
        print('perp_vector2: ', self.perp_vector2)

    def __repr__(self):
        """
        Useful for displaying an Orientation instance.

        :meta private:
        """
        s = 'Paicos Orientation instance\n'
        s += f'normal_vector: {self.normal_vector}\n'
        s += f'perp_vector1:  {self.perp_vector1}\n'
        s += f'perp_vector2:  {self.perp_vector2}'
        return s
