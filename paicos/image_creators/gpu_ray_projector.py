import numpy as np
import cupy as cp
from numba import cuda
import numba

from .image_creator import ImageCreator
from .. import util
from .. import settings
from .. import units

from ..trees.bvh_gpu import GpuBinaryTree
from ..trees.bvh_gpu import nearest_neighbor_device, nearest_neighbor_device_optimized


def get_blocks(size, threadsperblock):
    return (size + (threadsperblock - 1)) // threadsperblock


@cuda.jit(device=True, inline=True)
def rotate_point_around_center(point, tmp_point, center, rotation_matrix):
    """
    Rotate around center. Note that we overwrite point.
    """

    # Subtract center
    for ii in range(3):
        tmp_point[ii] = point[ii] - center[ii]
        point[ii] = 0.0

    # Rotate around center (matrix multiplication).
    for ii in range(3):
        for jj in range(3):
            point[ii] += rotation_matrix[ii, jj] * tmp_point[jj]

    # Add center back
    for ii in range(3):
        point[ii] = point[ii] + center[ii]


@cuda.jit
def trace_rays_voronoi(points, tree_parents, tree_children, tree_bounds, variable, hsml,
                       widths, center,
                       tree_scale_factor, tree_offsets, image, rotation_matrix, tol):

    ix, iy = cuda.grid(2)

    nx = image.shape[0]
    ny = image.shape[1]

    L = 21

    if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
        result = 0.0

        dx = widths[0] / nx
        dy = widths[1] / ny

        num_internal_nodes = tree_children.shape[0]

        # Initialize z and dz (in arepo code units)
        z = 0.0

        query_point = numba.cuda.local.array(3, numba.float64)
        tmp_point = numba.cuda.local.array(3, numba.float64)

        while z < widths[2]:

            # Query points in aligned coords
            query_point[0] = (center[0] - widths[0] / 2.0) + (ix + 0.5) * dx
            query_point[1] = (center[1] - widths[1] / 2.0) + (iy + 0.5) * dy
            query_point[2] = (center[2] - widths[2] / 2.0) + z

            # Rotate to simulation coords
            rotate_point_around_center(
                query_point, tmp_point, center, rotation_matrix)

            # Convert to the tree coordinates
            query_point[2] = (query_point[2] - tree_offsets[2]
                              ) * tree_scale_factor
            query_point[0] = (query_point[0] - tree_offsets[0]
                              ) * tree_scale_factor
            query_point[1] = (query_point[1] - tree_offsets[1]
                              ) * tree_scale_factor

            min_dist, min_index = nearest_neighbor_device(points, tree_parents, tree_children,
                                                          tree_bounds, query_point,
                                                          num_internal_nodes)

            if min_index == -1:
                min_dist, min_index = nearest_neighbor_device_optimized(points, tree_parents, tree_children,
                                                                        tree_bounds, query_point,
                                                                        num_internal_nodes, 2**L - 1.0)

            # Calculate dz
            # dz = tol * hsml[min_index]
            dz = tol * min_dist / tree_scale_factor

            # Update integral
            result = result + dz * variable[min_index]

            # Update position
            z = z + dz

        # Subtract the 'extra' stuff added in last iteration
        result = result - (z - widths[2]) * variable[min_index]
        # Set result in image array
        image[ix, iy] = result


@cuda.jit
def trace_rays_optimized(points, tree_parents, tree_children, tree_bounds, variable, hsml,
                         widths, center,
                         tree_scale_factor, tree_offsets, image, rotation_matrix, tol):

    ix, iy = cuda.grid(2)

    nx = image.shape[0]
    ny = image.shape[1]

    L = 21

    if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
        result = 0.0

        dx = widths[0] / nx
        dy = widths[1] / ny

        num_internal_nodes = tree_children.shape[0]

        # Initialize z and dz (in arepo code units)
        z = 0.0

        query_point = numba.cuda.local.array(3, numba.float64)
        tmp_point = numba.cuda.local.array(3, numba.float64)

        max_search_dist = (2**L - 1.0)

        while z < widths[2]:

            # Query points in aligned coords
            query_point[0] = (center[0] - widths[0] / 2.0) + (ix + 0.5) * dx
            query_point[1] = (center[1] - widths[1] / 2.0) + (iy + 0.5) * dy
            query_point[2] = (center[2] - widths[2] / 2.0) + z

            # Rotate to simulation coords
            rotate_point_around_center(
                query_point, tmp_point, center, rotation_matrix)

            # Convert to the tree coordinates
            query_point[2] = (query_point[2] - tree_offsets[2]
                              ) * tree_scale_factor
            query_point[0] = (query_point[0] - tree_offsets[0]
                              ) * tree_scale_factor
            query_point[1] = (query_point[1] - tree_offsets[1]
                              ) * tree_scale_factor

            min_dist, min_index = nearest_neighbor_device_optimized(points, tree_parents, tree_children,
                                                                    tree_bounds, query_point,
                                                                    num_internal_nodes, max_search_dist)

            # Calculate dz
            # dz = tol * hsml[min_index]
            dz = 2.0 * tol * min_dist / tree_scale_factor
            max_search_dist = 1.2 * (min_dist + dz * tree_scale_factor)

            # Update integral
            result = result + dz * variable[min_index]

            # Update position
            z = z + dz

        # Subtract the 'extra' stuff added in last iteration
        result = result - (z - widths[2]) * variable[min_index]
        # Set result in image array
        image[ix, iy] = result


class GpuRayProjector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane. This class works by raytracing the variable
    (i.e. by calculating a line integral along the line-of-sight).

    It only works on cuda-enabled GPUs.
    """

    @util.conditional_timer
    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, tol=0.25, threadsperblock=8,
                 do_pre_selection=False, timing=False):
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

        # err_msg = ('GpuRayProjector currently only works for parttype 0.' +
        #             ' This is because the GPU version  of BVH tree' +
        #             ' assumes that the query points are inside the bounding' +
        #             ' boxes of the leaves of the tree. This has been fixed for' +
        #             ' the CPU version. Please send Thomas an email if you see' +
        #             ' this error message.')

        # assert parttype == 0, err_msg

        # call the superclass constructor to initialize the ImageCreator class
        super().__init__(snap, center, widths, direction, npix=npix, parttype=parttype)

        parttype = self.parttype

        self.threadsperblock = threadsperblock

        self.do_pre_selection = do_pre_selection

        self.tol = tol

        # Calculate the smoothing length
        avail_list = (list(snap.keys()) + snap._auto_list)
        if f'{parttype}_Volume' in avail_list:
            self.hsml = 2.0 * np.cbrt((self.snap[f"{parttype}_Volume"])
                                      / (4.0 * np.pi / 3.0))
        elif f'{parttype}_SubfindHsml' in avail_list:
            self.hsml = self.snap[f'{parttype}_SubfindHsml']
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
                           + "you have turned on. If your GPU has enough memory "
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

            self._send_data_to_gpu()

            # We need to reconstruct the tree!
            self.tree = GpuBinaryTree(self.gpu_variables['pos'],
                                      self.gpu_variables['hsml'])
            del self.gpu_variables['pos']
            self.gpu_variables['hsml'] = self.gpu_variables['hsml'][self.tree.sort_index]
        # Send entirety of snapshot to GPU (if we have not already
        # done so). Always send small data with change in resolution etc
        else:
            if not self.has_do_region_selection_been_called:
                self._send_data_to_gpu()

                # Construct tree
                self.tree = GpuBinaryTree(self.gpu_variables['pos'],
                                          self.gpu_variables['hsml'])

                del self.gpu_variables['pos']
                self.gpu_variables['hsml'] = self.gpu_variables['hsml'][
                    self.tree.sort_index]

        # Always send small data
        self._send_small_data_to_gpu()

    def _send_data_to_gpu(self):
        self.gpu_variables = {}
        if settings.use_units:
            self.gpu_variables['pos'] = cp.array(self.pos.value)
            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
        else:
            self.gpu_variables['pos'] = cp.array(self.pos)
            self.gpu_variables['hsml'] = cp.array(self.hsml)

        self._send_small_data_to_gpu()

    def _send_small_data_to_gpu(self):

        self.gpu_variables['rotation_matrix'] = cp.array(
            self.orientation.rotation_matrix)

        if settings.use_units:
            self.gpu_variables['widths'] = cp.array(self.widths.value)
            self.gpu_variables['center'] = cp.array(self.center.value)
        else:
            self.gpu_variables['widths'] = cp.array(self.widths)
            self.gpu_variables['center'] = cp.array(self.center)

    def _gpu_project(self, variable_str, additive):
        """
        Private method for projecting using cuda code
        """
        gpu_vars = self.gpu_variables
        rotation_matrix = gpu_vars['rotation_matrix']
        widths = gpu_vars['widths']
        center = gpu_vars['center']
        hsml = gpu_vars['hsml']
        nx = self.npix_width
        ny = self.npix_height

        if additive:
            variable = gpu_vars[variable_str] / gpu_vars[f'{self.parttype}_Volume']
        else:
            variable = gpu_vars[variable_str]

        tree_scale_factor = self.tree.conversion_factor
        tree_offsets = self.tree.off_sets

        image = cp.zeros((nx, ny))

        blocks_x = get_blocks(nx, self.threadsperblock)
        blocks_y = get_blocks(ny, self.threadsperblock)
        btuple = (blocks_x, blocks_y)
        ttuple = (self.threadsperblock, self.threadsperblock)
        if self.parttype == 0:
            trace_rays = trace_rays_voronoi
        else:
            trace_rays = trace_rays_optimized

        trace_rays[btuple, ttuple](self.tree._pos, self.tree.parents, self.tree.children,
                                   self.tree.bounds, variable, hsml, widths, center,
                                   tree_scale_factor, tree_offsets, image,
                                   rotation_matrix, self.tol)

        projection = cp.asnumpy(image)
        return projection

    def _send_variable_to_gpu(self, variable, gpu_key='projection_variable'):
        if isinstance(variable, str):
            variable_str = str(variable)
            err_msg = 'projector uses a different parttype'
            assert int(variable[0]) == self.parttype, err_msg
            variable = self.snap[variable]
        else:
            variable_str = 'projection_variable'
            if not isinstance(variable, np.ndarray):
                raise RuntimeError('Unexpected type for variable')

        assert len(variable.shape) == 1, 'only scalars can be projected'

        # Select same part of array that the projector has selected
        if self.do_pre_selection:
            # TODO: Check that this is not applied more than once...
            variable = variable[self.index]

        if variable_str in self.gpu_variables and variable_str != gpu_key:
            pass
        else:
            # Send variable to gpu
            if settings.use_units:
                self.gpu_variables[variable_str] = cp.array(variable.value)
            else:
                self.gpu_variables[variable_str] = cp.array(variable)

            # Sort the variable according to Morton code sorting
            self.gpu_variables[variable_str] = self.gpu_variables[variable_str][
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

        variable_str, unit_quantity = self._send_variable_to_gpu(variable)
        if additive:
            _, vol_unit_quantity = self._send_variable_to_gpu(f'{self.parttype}_Volume')

        # Do the projection
        projection = self._gpu_project(variable_str, additive)

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

    def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        self.release_gpu_memory()

    def release_gpu_memory(self):
        if hasattr(self, 'gpu_variables'):
            for key in list(self.gpu_variables):
                del self.gpu_variables[key]
            del self.gpu_variables
        if hasattr(self, 'tree'):
            self.tree.release_gpu_memory()
            del self.tree

        cp._default_memory_pool.free_all_blocks()
