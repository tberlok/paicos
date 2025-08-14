import numpy as np
import numba

from .image_creator import ImageCreator
from .. import util
from .. import settings
from .. import units

from ..trees.bvh_cpu import BinaryTree
from ..trees.bvh_cpu import nearest_neighbor_cpu, nearest_neighbor_cpu_voronoi, nearest_neighbor_cpu_optimized

from ..trees.bvh_cpu import leading_zeros_cython_for_numba


@numba.njit(inline='always')
def part1by2_64(n):
    n &= 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8)) & 0x100f00f00f00f00f
    n = (n | (n << 4)) & 0x10c30c30c30c30c3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


@numba.njit(inline='always')
def unpart1by2_64(n):
    n = numba.uint64(n)
    n &= 0x1249249249249249
    n = (n ^ (n >> 2)) & 0x10c30c30c30c30c3
    n = (n ^ (n >> 4)) & 0x100f00f00f00f00f
    n = (n ^ (n >> 8)) & 0x1f0000ff0000ff
    n = (n ^ (n >> 16)) & 0x1f00000000ffff
    n = (n ^ (n >> 32)) & 0x1fffff
    return n


@numba.njit(inline='always')
def encode64(x, y, z):
    xx = part1by2_64(x)
    yy = part1by2_64(y)
    zz = part1by2_64(z)
    return xx * 4 + yy * 2 + zz


@numba.njit(parallel=True)
def trace_rays_cpu(points, tree_parents, tree_children, tree_bounds, variable, hsml,
                   widths, center, tree_scale_factor, tree_offsets, image, rotation_matrix, tol):

    nx = image.shape[0]
    ny = image.shape[1]
    dx = widths[0] / nx
    dy = widths[1] / ny
    num_internal_nodes = tree_children.shape[0]

    for ix in numba.prange(nx):  # pylint: disable=not-an-iterable
        for iy in range(ny):
            result = 0.0
            z = 0.0
            query_point = np.empty(3, dtype=np.float64)

            while z < widths[2]:
                # Compute query point in aligned coords
                query_point[0] = (center[0] - widths[0]
                                  / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1]
                                  / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center
                query_point = np.dot(
                    rotation_matrix, query_point - center) + center

                # Convert to tree coordinates
                query_point[0] = (
                    query_point[0] - tree_offsets[0]) * tree_scale_factor
                query_point[1] = (
                    query_point[1] - tree_offsets[1]) * tree_scale_factor
                query_point[2] = (
                    query_point[2] - tree_offsets[2]) * tree_scale_factor

                # Nearest neighbor search in tree coordinates
                min_dist, min_index = nearest_neighbor_cpu(points, tree_parents, tree_children,
                                                           tree_bounds, query_point,
                                                           num_internal_nodes)

                # Adaptive step in Arepo units
                # dz = tol * hsml[min_index]
                dz = tol * min_dist / tree_scale_factor
                # if hsml[min_index] < min_dist:
                # print(hsml[min_index], min_dist, min_dist/tree_scale_factor)
                result += dz * variable[min_index]
                z += dz

            # Correct overshoot
            result -= (z - widths[2]) * variable[min_index]
            image[ix, iy] = result


@numba.njit(parallel=True)
def trace_rays_cpu_voronoi(points, tree_parents, tree_children, tree_bounds, variable, hsml,
                           widths, center, tree_scale_factor, tree_offsets, image, rotation_matrix, tol):

    nx = image.shape[0]
    ny = image.shape[1]
    dx = widths[0] / nx
    dy = widths[1] / ny
    num_internal_nodes = tree_children.shape[0]
    L = 21

    for ix in numba.prange(nx):  # pylint: disable=not-an-iterable
        for iy in range(ny):
            result = 0.0
            z = 0.0
            query_point = np.empty(3, dtype=np.float64)

            while z < widths[2]:
                # Compute query point in aligned coords
                query_point[0] = (center[0] - widths[0]
                                  / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1]
                                  / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center
                query_point = np.dot(
                    rotation_matrix, query_point - center) + center

                # Convert to tree coordinates
                query_point[0] = (
                    query_point[0] - tree_offsets[0]) * tree_scale_factor
                query_point[1] = (
                    query_point[1] - tree_offsets[1]) * tree_scale_factor
                query_point[2] = (
                    query_point[2] - tree_offsets[2]) * tree_scale_factor

                # Nearest neighbor search in tree coordinates
                min_dist, min_index = nearest_neighbor_cpu_voronoi(points, tree_parents, tree_children,
                                                                   tree_bounds, query_point,
                                                                   num_internal_nodes)
                if min_index == -1:
                    min_dist, min_index = nearest_neighbor_cpu_optimized(points, tree_parents, tree_children,
                                                                         tree_bounds, query_point,
                                                                         num_internal_nodes, (2**L - 1.0))
                # Adaptive step in Arepo units
                dz = tol * hsml[min_index]
                result += dz * variable[min_index]
                z += dz

            # Correct overshoot
            result -= (z - widths[2]) * variable[min_index]
            image[ix, iy] = result


@numba.njit(parallel=True)
def trace_rays_several_variables_cpu_voronoi(points, tree_parents, tree_children, tree_bounds,
                                             variables, hsml, widths, center,
                                             tree_scale_factor, tree_offsets, images,
                                             rotation_matrix, tol):

    nx = images.shape[0]
    ny = images.shape[1]
    nvars = images.shape[2]
    dx = widths[0] / nx
    dy = widths[1] / ny
    num_internal_nodes = tree_children.shape[0]
    L = 21

    for ix in numba.prange(nx):  # pylint: disable=not-an-iterable
        for iy in range(ny):
            result = np.zeros(nvars, dtype=variables.dtype)
            z = 0.0
            query_point = np.empty(3, dtype=np.float64)

            while z < widths[2]:
                # Compute query point in aligned coords
                query_point[0] = (center[0] - widths[0]
                                  / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1]
                                  / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center
                query_point = np.dot(
                    rotation_matrix, query_point - center) + center

                # Convert to tree coordinates
                query_point[0] = (
                    query_point[0] - tree_offsets[0]) * tree_scale_factor
                query_point[1] = (
                    query_point[1] - tree_offsets[1]) * tree_scale_factor
                query_point[2] = (
                    query_point[2] - tree_offsets[2]) * tree_scale_factor

                # Nearest neighbor search in tree coordinates
                min_dist, min_index = nearest_neighbor_cpu_voronoi(points, tree_parents, tree_children,
                                                                   tree_bounds, query_point,
                                                                   num_internal_nodes)
                if min_index == -1:
                    min_dist, min_index = nearest_neighbor_cpu_optimized(points, tree_parents, tree_children,
                                                                         tree_bounds, query_point,
                                                                         num_internal_nodes, (2**L - 1.0))

                # Adaptive step in Arepo units
                dz = tol * hsml[min_index]

                for k in range(nvars):
                    result[k] += dz * variables[min_index, k]
                z += dz

            # Correct overshoot
            for k in range(nvars):
                result[k] -= (z - widths[2]) * variables[min_index, k]
                images[ix, iy, k] = result[k]


@numba.njit(parallel=True)
def trace_rays_cpu_optimized(points, tree_parents, tree_children, tree_bounds, variable, hsml,
                             widths, center, tree_scale_factor, tree_offsets, image, rotation_matrix, tol):

    nx = image.shape[0]
    ny = image.shape[1]
    dx = widths[0] / nx
    dy = widths[1] / ny
    num_internal_nodes = tree_children.shape[0]

    L = 21

    for ix in numba.prange(nx):  # pylint: disable=not-an-iterable
        for iy in range(ny):
            result = 0.0
            z = 0.0
            query_point = np.empty(3, dtype=np.float64)

            max_search_dist = (2**L - 1.0)

            while z < widths[2]:
                # Compute query point in aligned coords
                query_point[0] = (center[0] - widths[0]
                                  / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1]
                                  / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center
                query_point = np.dot(
                    rotation_matrix, query_point - center) + center

                # Convert to tree coordinates
                query_point[0] = (
                    query_point[0] - tree_offsets[0]) * tree_scale_factor
                query_point[1] = (
                    query_point[1] - tree_offsets[1]) * tree_scale_factor
                query_point[2] = (
                    query_point[2] - tree_offsets[2]) * tree_scale_factor

                # Nearest neighbor search in tree coordinates
                min_dist, min_index = nearest_neighbor_cpu_optimized(points, tree_parents, tree_children,
                                                                     tree_bounds, query_point,
                                                                     num_internal_nodes, max_search_dist)
                # Adaptive step in Arepo units
                dz = 2.0 * tol * min_dist / tree_scale_factor
                max_search_dist = 1.1 * (min_dist + dz * tree_scale_factor)
                result += dz * variable[min_index]
                z += dz

            # Correct overshoot
            result -= (z - widths[2]) * variable[min_index]
            image[ix, iy] = result


@numba.njit(parallel=True)
def trace_rays_several_variables_cpu_optimized(points, tree_parents, tree_children, tree_bounds,
                                               variables, hsml, widths, center,
                                               tree_scale_factor, tree_offsets, images,
                                               rotation_matrix, tol):

    nx = images.shape[0]
    ny = images.shape[1]
    nvars = images.shape[2]
    dx = widths[0] / nx
    dy = widths[1] / ny
    num_internal_nodes = tree_children.shape[0]

    L = 21

    for ix in numba.prange(nx):  # pylint: disable=not-an-iterable
        for iy in range(ny):
            result = np.zeros(nvars, dtype=variables.dtype)
            z = 0.0
            query_point = np.empty(3, dtype=np.float64)

            max_search_dist = (2**L - 1.0)

            while z < widths[2]:
                # Compute query point in aligned coords
                query_point[0] = (center[0] - widths[0]
                                  / 2.0) + (ix + 0.5) * dx
                query_point[1] = (center[1] - widths[1]
                                  / 2.0) + (iy + 0.5) * dy
                query_point[2] = (center[2] - widths[2] / 2.0) + z

                # Rotate point around center
                query_point = np.dot(
                    rotation_matrix, query_point - center) + center

                # Convert to tree coordinates
                query_point[0] = (
                    query_point[0] - tree_offsets[0]) * tree_scale_factor
                query_point[1] = (
                    query_point[1] - tree_offsets[1]) * tree_scale_factor
                query_point[2] = (
                    query_point[2] - tree_offsets[2]) * tree_scale_factor

                # Nearest neighbor search in tree coordinates
                min_dist, min_index = nearest_neighbor_cpu_optimized(points, tree_parents, tree_children,
                                                                     tree_bounds, query_point,
                                                                     num_internal_nodes, max_search_dist)
                # Adaptive step in Arepo units
                dz = 2.0 * tol * min_dist / tree_scale_factor
                max_search_dist = 1.1 * (min_dist + dz * tree_scale_factor)

                for k in range(nvars):
                    result[k] += dz * variables[min_index, k]
                z += dz

            # Correct overshoot
            for k in range(nvars):
                result[k] -= (z - widths[2]) * variables[min_index, k]
                images[ix, iy, k] = result[k]


class RayProjector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane. This class works by raytracing the variable
    (i.e. by calculating a line integral along the line-of-sight).

    This is crudely made CPU-version which was made by adapting
    the corresponding GPU code back to CPU.
    """

    @util.conditional_timer
    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, tol=0.25, do_pre_selection=False,
                 use_numba=True, verbose=False, timing=False):
        """
        Initialize the Projector class.

        Parameters
        ----------
        snap : Snapshot
            A snapshot object of Snapshot class from paicos package.

        center : numpy array
            Center of the region on which projection is to be done, e.g.
            center = [x_c, y_c, z_c].

        widths : numpy array
            Widths of the region on which projection is to be done,
            e.g.m widths=[width_x, width_y, width_z].

        direction : str
            Direction of the projection, e.g. 'x', 'y' or 'z',
            or a Paicos Orientation class instance.

        npix : int, optional
            Number of pixels in the horizontal direction of the image,
            by default 512.

        parttype : int, optional
            Number of the particle type to project, by default gas (PartType 0).

        """

        # call the superclass constructor to initialize the ImageCreator class
        super().__init__(snap, center, widths, direction, npix=npix, parttype=parttype)

        parttype = self.parttype

        self.do_pre_selection = do_pre_selection

        self.tol = tol

        self.use_numba = use_numba

        self.verbose = verbose

        # Calculate the smoothing length
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            self.hsml = 1.2 * 2.0 * np.cbrt((self.snap[f"{parttype}_Volume"])
                                            / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            self.hsml = self.snap[f'{parttype}_SubfindHsml']
        else:
            if self.parttype != 0:
                self.hsml = np.zeros_like(self.snap[f'{parttype}_Coordinates'])
            else:
                raise RuntimeError(
                    'There is no smoothing length or volume for the projector')

        self.pos = self.snap[f'{self.parttype}_Coordinates']

        if settings.use_units:
            self.hsml = self.hsml.to(self.pos.unit)

        # Call selection
        self.has_do_region_selection_been_called = False
        self._do_region_selection()
        self.has_do_region_selection_been_called = True

    def _do_region_selection(self):

        self.do_unit_consistency_check()

        center = self.center
        widths = self.widths
        snap = self.snap
        parttype = self.parttype

        if self.has_do_region_selection_been_called:
            if self.do_pre_selection:
                import warnings
                err_msg = ("It looks like you are changing projector "
                           + " properties after the fact, i.e. changing widths "
                           + "center, orientation, resolution etc. This might be "
                           + "slow with with the option do_pre_selection, which "
                           + "you have turned on. If you have enough memory "
                           + "then it is probably better to set do_pre_selection "
                           + "to False.")
                warnings.warn(err_msg)

        # Send subset of snapshot to GPU
        if self.do_pre_selection:
            # get the index of the region of projection
            if self.direction != 'orientation':
                get_index = util.get_index_of_cubic_region_plus_thin_layer
                self.index = get_index(self.snap[f"{parttype}_Coordinates"],
                                       center, widths, self.hsml, snap.box)
            else:
                get_index = util.get_index_of_rotated_cubic_region_plus_thin_layer
                self.index = get_index(snap[f"{parttype}_Coordinates"],
                                       center, widths, self.hsml, snap.box,
                                       self.orientation)

            self.hsml = self.hsml[self.index]
            self.pos = self.pos[self.index]

            self._send_data_to_tree()

            # We need to reconstruct the tree!
            self.tree = BinaryTree(self.tree_variables['pos'],
                                   self.tree_variables['hsml'], self.verbose)
            del self.tree_variables['pos']
            self.tree_variables['hsml'] = self.tree_variables['hsml'][self.tree.sort_index]
        # Send entirety of snapshot to GPU (if we have not already
        # done so). Always send small data with change in resolution etc
        else:
            if not self.has_do_region_selection_been_called:
                self._send_data_to_tree()

                # Construct tree
                self.tree = BinaryTree(self.tree_variables['pos'],
                                       self.tree_variables['hsml'], self.verbose)

                del self.tree_variables['pos']
                self.tree_variables['hsml'] = self.tree_variables['hsml'][
                    self.tree.sort_index]

        # Always send small data
        self._send_small_data_to_tree()

    def _send_data_to_tree(self):
        self.tree_variables = {}
        if settings.use_units:
            self.tree_variables['pos'] = np.array(self.pos.value)
            self.tree_variables['hsml'] = np.array(self.hsml.value)
        else:
            self.tree_variables['pos'] = np.array(self.pos)
            self.tree_variables['hsml'] = np.array(self.hsml)

        self._send_small_data_to_tree()

    def _send_small_data_to_tree(self):

        self.tree_variables['rotation_matrix'] = np.array(
            self.orientation.rotation_matrix)

        if settings.use_units:
            self.tree_variables['widths'] = np.array(self.widths.value)
            self.tree_variables['center'] = np.array(self.center.value)
        else:
            self.tree_variables['widths'] = np.array(self.widths)
            self.tree_variables['center'] = np.array(self.center)

    def _tree_project(self, variable_str, additive):
        """
        Private method for projecting using cuda code
        """
        tree_vars = self.tree_variables
        rotation_matrix = tree_vars['rotation_matrix']
        widths = tree_vars['widths']
        center = tree_vars['center']
        hsml = tree_vars['hsml']
        nx = self.npix_width
        ny = self.npix_height

        if additive:
            variable = tree_vars[variable_str] / tree_vars[f'{self.parttype}_Volume']
        else:
            variable = tree_vars[variable_str]

        tree_scale_factor = self.tree.conversion_factor
        tree_offsets = self.tree.off_sets

        image = np.zeros((nx, ny))

        if self.use_numba:

            if self.parttype == 0:
                trace_rays = trace_rays_cpu_voronoi
            else:
                trace_rays = trace_rays_cpu_optimized

            trace_rays(self.tree._pos, self.tree.parents, self.tree.children,
                       self.tree.bounds, variable, hsml, widths, center,
                       tree_scale_factor, tree_offsets, image,
                       rotation_matrix, self.tol)
        else:
            import warnings
            warnings.warn(
                "Cython ray tracer slower and less well tested than Numba!")
            if self.parttype == 0:
                from ..cython.ray_tracer import trace_rays_cpu_voronoi as trace_rays_cpu_cython
            else:
                from ..cython.ray_tracer import trace_rays_cpu as trace_rays_cpu_cython

            trace_rays_cpu_cython(self.tree._pos, self.tree.parents, self.tree.children,
                                  self.tree.bounds, variable, hsml, widths, center,
                                  tree_scale_factor, tree_offsets, image,
                                  rotation_matrix, self.tol, 1)

        return image

    def _tree_project_several_variables(self, variable_strs, additives):
        """
        Private method for projecting several variables in one tree traversal.

        Parameters
        ----------
        variable_strs : list of str
            Keys in `self.tree_variables` for each variable to project.
        additives : list of bool
            Whether to use additive projection for each variable.

        Returns
        -------
        numpy.ndarray
            Array of shape (nx, ny, n_variables) with the projected results.
        """
        tree_vars = self.tree_variables
        rotation_matrix = tree_vars['rotation_matrix']
        widths = tree_vars['widths']
        center = tree_vars['center']
        hsml = tree_vars['hsml']
        nx = self.npix_width
        ny = self.npix_height

        # Build (N, n_variables) array for particle variables
        var_list = []
        for var_str, additive in zip(variable_strs, additives):
            if additive:
                var_arr = tree_vars[var_str] / tree_vars[f'{self.parttype}_Volume']
            else:
                var_arr = tree_vars[var_str]
            var_list.append(var_arr)

        variables = np.column_stack(var_list)  # shape: (N, n_variables)

        tree_scale_factor = self.tree.conversion_factor
        tree_offsets = self.tree.off_sets

        # Allocate output cube (nx, ny, n_variables)
        images = np.zeros((nx, ny, len(variable_strs)), dtype=variables.dtype)

        if self.use_numba:
            if self.parttype == 0:
                trace_rays = trace_rays_several_variables_cpu_voronoi
            else:
                trace_rays = trace_rays_several_variables_cpu_optimized

            trace_rays(self.tree._pos, self.tree.parents, self.tree.children,
                       self.tree.bounds, variables, hsml, widths, center,
                       tree_scale_factor, tree_offsets, images,
                       rotation_matrix, self.tol)
        else:
            raise RuntimeError("only implemented for numba")

        return images

    def _send_variable_to_tree(self, variable, tree_key='projection_variable'):
        if isinstance(variable, str):
            variable_str = str(variable)
            err_msg = 'projector uses a different parttype'
            assert int(variable[0]) == self.parttype, err_msg
            variable = self.snap[variable]
        else:
            variable_str = tree_key
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars can be projected'

        # Select same part of array that the projector has selected
        if self.do_pre_selection:
            # TODO: Check that this is not applied more than once...
            variable = variable[self.index]

        if variable_str in self.tree_variables and variable_str != tree_key:
            pass
        else:
            # Send variable to gpu
            if settings.use_units:
                self.tree_variables[variable_str] = np.array(variable.value)
            else:
                self.tree_variables[variable_str] = np.array(variable)

            # Sort the variable according to Morton code sorting
            self.tree_variables[variable_str] = self.tree_variables[variable_str][
                self.tree.sort_index]

        if isinstance(variable, units.PaicosQuantity):
            unit_quantity = variable.unit_quantity
        else:
            unit_quantity = None

        return variable_str, unit_quantity

    @util.conditional_timer
    def project_variable(self, variable, additive=False, timing=False):
        """
        projects a given variable onto a 2D plane.

        Parameters
        ----------
        variable : str, function, numpy array
            variable, it can be passed as string or an array

        Returns
        -------
        numpy array
            The image of the projected variable
        """

        # This calls _do_region_selection if resolution, Orientation,
        # widths or center changed
        self._check_if_properties_changed()

        variable_str, unit_quantity = self._send_variable_to_tree(variable)
        if additive:
            _, vol_unit_quantity = self._send_variable_to_tree(f'{self.parttype}_Volume')

        # Do the projection
        projection = self._tree_project(variable_str, additive)

        # Transpose
        projection = projection.T

        assert projection.shape[0] == self.npix_height
        assert projection.shape[1] == self.npix_width

        if unit_quantity is not None:
            unit_length = self.snap['0_Coordinates'].uq
            projection = projection * unit_quantity * unit_length
            if additive:
                projection = projection / vol_unit_quantity

        if additive:
            return projection
        else:
            return projection / self.depth

    @util.conditional_timer
    def project_variables(self, variables, additives, timing=False):
        """
        Projects multiple variables onto a 2D plane in a single tree traversal.

        This probably offers a speed benefit compared to calling
        project_variable several times.

        Parameters
        ----------
        variables : list of str, function, or numpy array
            Variables to project.
        additives : list of bool
            Whether to use additive projection for each variable.

        Returns
        -------
        list
            List of 2D arrays, each being the projection of the corresponding variable.
        """
        assert len(variables) == len(additives), \
            "Length of `variables` and `additives` must match."

        self._check_if_properties_changed()

        tree_keys = []
        unit_quantities = []
        vol_unit_quantities = []
        is_additive_flags = []

        for i, (variable, additive) in enumerate(zip(variables, additives)):
            # Assign unique tree key for this projection variable
            # (not used for string variables!)
            tree_key = f'projection_variable{i}'

            variable_str, unit_quantity = self._send_variable_to_tree(
                variable, tree_key=tree_key
            )
            tree_keys.append(variable_str)
            unit_quantities.append(unit_quantity)
            is_additive_flags.append(additive)

            if additive:
                _, vol_unit_quantity = self._send_variable_to_tree(
                    f'{self.parttype}_Volume'
                )
            else:
                vol_unit_quantity = None
            vol_unit_quantities.append(vol_unit_quantity)

        # Perform projection in one tree traversal
        projection_cube = self._tree_project_several_variables(
            tree_keys, is_additive_flags)
        # projection_cube.shape = (nx, ny, n_variables)

        results = []
        for i in range(len(variables)):
            # transpose for output convention
            proj_2d = projection_cube[:, :, i].T

            assert proj_2d.shape[0] == self.npix_height
            assert proj_2d.shape[1] == self.npix_width

            unit_quantity = unit_quantities[i]
            vol_unit_quantity = vol_unit_quantities[i]
            additive = is_additive_flags[i]

            if unit_quantity is not None:
                unit_length = self.snap['0_Coordinates'].uq
                proj_2d = proj_2d * unit_quantity * unit_length
                if additive:
                    proj_2d = proj_2d / vol_unit_quantity

            if not additive:
                proj_2d = proj_2d / self.depth

            results.append(proj_2d)

        return results

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        self.release_tree_memory()

    def release_tree_memory(self):
        if hasattr(self, 'tree_variables'):
            for key in list(self.tree_variables):
                del self.tree_variables[key]
            del self.tree_variables
        if hasattr(self, 'tree'):
            # self.tree.release_tree_memory()
            del self.tree
