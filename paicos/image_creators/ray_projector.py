import numpy as np
import numba

from .image_creator import ImageCreator
from .. import util
from .. import settings
from .. import units

from ..trees.bvh_cpu import BinaryTree
from .cpu_ray_cython import trace_rays


class RayProjector(ImageCreator):
    """
    A class that allows creating an image of a given variable by projecting
    it onto a 2D plane. This class works by raytracing the variable
    (i.e. by calculating a line integral along the line-of-sight).
    """

    def __init__(self, snap, center, widths, direction,
                 npix=512, parttype=0, tol=0.25,
                 do_pre_selection=True):
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
                           + "you have turned on. If you have enough memory "
                           + "then it is probably better to set do_pre_selection "
                           + "to False.")
                warnings.warn(err_msg)

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

            self._send_data_to_cpu()

            # We need to reconstruct the tree!
            self.tree = BinaryTree(self.cpu_variables['pos'],
                                   self.cpu_variables['hsml'])
            del self.cpu_variables['pos']
            self.cpu_variables['hsml'] = self.cpu_variables['hsml'][self.tree.sort_index]
        # Send entirety of snapshot to GPU (if we have not already
        # done so). Always send small data with change in resolution etc
        else:
            if not self.has_do_region_selection_been_called:
                self._send_data_to_cpu()

                # Construct tree
                self.tree = BinaryTree(self.cpu_variables['pos'],
                                       self.cpu_variables['hsml'])

                del self.cpu_variables['pos']
                self.cpu_variables['hsml'] = self.cpu_variables['hsml'][
                    self.tree.sort_index]

        # Always send small data
        self._send_small_data_to_cpu()

    def _send_data_to_cpu(self):
        self.cpu_variables = {}
        if settings.use_units:
            self.cpu_variables['pos'] = np.array(self.pos.value)
            self.cpu_variables['hsml'] = np.array(self.hsml.value)
        else:
            self.cpu_variables['pos'] = np.array(self.pos)
            self.cpu_variables['hsml'] = np.array(self.hsml)

        self._send_small_data_to_cpu()

    def _send_small_data_to_cpu(self):

        self.cpu_variables['rotation_matrix'] = np.array(
            self.orientation.rotation_matrix)

        if settings.use_units:
            self.cpu_variables['widths'] = np.array(self.widths.value)
            self.cpu_variables['center'] = np.array(self.center.value)
        else:
            self.cpu_variables['widths'] = np.array(self.widths)
            self.cpu_variables['center'] = np.array(self.center)

    def _cpu_project(self, variable_str):
        """
        Private method for projecting using cuda code
        """
        rotation_matrix = self.cpu_variables['rotation_matrix']
        widths = self.cpu_variables['widths']
        center = self.cpu_variables['center']
        hsml = self.cpu_variables['hsml']
        nx = self.npix_width
        ny = self.npix_height
        variable = self.cpu_variables[variable_str]

        tree_scale_factor = self.tree.conversion_factor
        tree_offsets = self.tree.off_sets

        image = np.zeros((nx, ny))

        trace_rays(self.tree._pos, self.tree.parents, self.tree.children,
                   self.tree.bounds, variable, hsml, widths, center,
                   tree_scale_factor, tree_offsets, image,
                   rotation_matrix, self.tol, settings.numthreads)

        projection = np.array(image)
        return projection

    def project_variable(self, variable, additive=False):
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

        if additive:
            err_msg = "GPU ray tracer does not yet support additive=True"
            raise RuntimeError(err_msg)

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
            variable = variable[self.index]

        if variable_str in self.cpu_variables and variable_str != 'projection_variable':
            pass
        else:
            # Send variable to cpu
            if settings.use_units:
                self.cpu_variables[variable_str] = np.array(variable.value)
            else:
                self.cpu_variables[variable_str] = np.array(variable)

            # Sort the variable according to Morton code sorting
            self.cpu_variables[variable_str] = self.cpu_variables[variable_str][
                self.tree.sort_index]

        # Do the projection
        projection = self._cpu_project(variable_str)

        # Transpose
        projection = projection.T

        assert projection.shape[0] == self.npix_height
        assert projection.shape[1] == self.npix_width

        if isinstance(variable, units.PaicosQuantity):
            unit_length = self.snap['0_Coordinates'].uq
            projection = projection * variable.unit_quantity * unit_length

        return projection / self.depth

    # def __del__(self):
        """
        Clean up like this? Not sure it is needed...
        """
        # del self.cpu_variables
        # del self.tree
        # cp._default_memory_pool.free_all_blocks()
