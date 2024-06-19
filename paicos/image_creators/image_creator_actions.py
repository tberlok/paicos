import numpy as np
from ..writers.arepo_image import ImageWriter
from ..readers.paicos_readers import ImageReader
import os


def find_angle_subdivisions(angle, max_angle):
    kk = 1
    angle = abs(angle)
    new_angle = angle / kk
    while new_angle > max_angle:
        kk += 1
        new_angle = angle / kk

    return kk


def find_zoom_sub_divisions(zoom, zoom_max):
    kk = 1
    if zoom < 1:
        zoom = 1 / zoom
    new_zoom = zoom**(1 / kk)
    while new_zoom > zoom_max:
        kk += 1
        new_zoom = zoom**(1 / kk)

    return kk


class Actions:
    """
    The purpose of this class is to simplify the steps
    that are usually involved in creating animations
    and/or interactive widgets.

    This is very much work in progress and subject
    to change. Send Thomas an email if you would like
    a working example.
    """
    def __init__(self, snapnum=None, image_creator=None, dry_run=True):

        self.dry_run = dry_run

        self.first_call = True
        if snapnum is None:
            if image_creator is not None:
                self.image_creator = image_creator
            else:
                msg = ("self.image_creator not set, you'll "
                       + "nedd to do this manually or read "
                       + "a log file using Actions.read_log")
                print(msg)
        else:
            self.make_image_creator(snapnum)
        self.first_call = False

        self.logging = False
        self.zoom_max = 10000000.
        self.max_angle = 360.

    def make_image_creator(self, snapnum):
        """

        TODO: Add example implementation in the docstring of
        of this method.
        """
        err_msg = ("You need to implement this yourself.\n"
                   + "Please see the docstring of this method")
        raise RuntimeError(err_msg)

    def mylogger(self, string, line_ending='\n', mode='a', command=True):
        outfolder = self.outfolder
        basename = self.basename
        with open(f'{outfolder}/{basename}.log', mode) as f:
            f.write(string + line_ending)

        if self.make_waypoints and command:
            self.frame_num += 1
            waypoint_folder = f'{self.outfolder}/{self.basename}_waypoints'
            self._make_waypoint(waypoint_folder, self.basename, self.frame_num)

    def create_log(self, outfolder='.', basename='image'):

        if outfolder[-1] != '/':
            outfolder += '/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        self.make_waypoints = True
        self.frame_num = 0
        self.outfolder = outfolder
        self.basename = basename
        self.mylogger('# Paicos image log file', mode='w', command=False)
        self.mylogger(outfolder, command=False)
        self.mylogger(basename, command=False)

        image_file = ImageWriter(self.image_creator, basedir=outfolder,
                                 basename=f'{basename}_logfile_start')
        image_file.finalize()
        self.mylogger(image_file.filename, command=False)
        self.logging = True

        if self.make_waypoints:
            waypoint_folder = f'{self.outfolder}{self.basename}_waypoints/'
            self._make_waypoint(waypoint_folder, self.basename, 0)

    def read_log(self, filename=None, outfolder='.', basename='image'):
        self.make_waypoints = False
        if filename is None:
            filename = f'{outfolder}/{basename}.log'
        with open(filename, 'r') as f:
            lines = f.readlines()
        self.outfolder = outfolder = lines[1].strip()
        self.basename = basename = lines[2].strip()
        self.logfile_start = lines[3].strip()
        self.lines = lines
        self.commands = self.lines[4:]
        self.frame_num = 0
        self.line_index = 4
        self.num_steps = len(self.lines) - self.line_index
        self.image_creator = ImageReader(self.logfile_start)
        self.first_call = False
        self.make_image_creator(self.image_creator.snapnum)

        waypoint_folder = f'{self.outfolder}/{self.basename}_waypoints/'
        if os.path.exists(waypoint_folder):
            import glob
            files = glob.glob(f'{waypoint_folder}*.hdf5')
            framenums = [int(file.split('_')[-2]) for file in files]
            snapnums = [int(file[-8:-5]) for file in files]
            index = np.argsort(framenums)
            files = [files[ii] for ii in index]
            self.waypoints = {}
            framenums = np.array(framenums)[index]
            snapnums = np.array(snapnums)[index]
            for ii, frame_num in enumerate(framenums):
                self.waypoints[frame_num] = files[ii]

    def resume_from_waypoint(self, frame_num):
        self.frame_num = frame_num
        self.line_index = self.frame_num + 4

        self.image_creator = ImageReader(self.waypoints[frame_num])
        self.first_call = False
        self.make_image_creator(self.image_creator.snapnum)

    def resume_from_image(self, filename, frame_num):
        raise RuntimeError('not implemented, try resume_from_waypoint instead')

    @property
    def next_action(self):
        return self.lines[self.line_index].strip()

    @property
    def previous_action(self):
        if self.line_index == 4:
            return None
        else:
            return self.lines[self.line_index - 1].strip()

    def step(self, verbose=False):
        line = self.next_action
        parts = line.split(',')
        command = parts[0]

        im_creator = self.image_creator

        if verbose:
            print(line)
        if command == 'center_on_sub' or command == 'center_on_group':
            # cx, cy, cz = float(parts[2]), float(parts[3]), float(parts[4])
            # new_center = np.array([cx, cy, cz]) * im_creator.center.uq
            raise RuntimeError("not implemented")
            # self.reset_center(new_center)
        elif command == 'zoom':
            zoom_fac = float(parts[1])
            if zoom_fac > self.zoom_max or (1 / zoom_fac) > self.zoom_max:
                kk = find_zoom_sub_divisions(zoom_fac, self.zoom_max)
                new_zoom_fac = zoom_fac**(1 / kk)
                for _ in range(kk):
                    self.zoom(new_zoom_fac)
            else:
                self.zoom(zoom_fac)
        elif command == 'width':
            self.width(float(parts[1]))
        elif command == 'height':
            self.height(float(parts[1]))
        elif command == 'depth':
            self.depth(float(parts[1]))
        elif command == 'rotate_around_perp_vector1':
            angle = float(parts[1])
            if abs(angle) > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_perp_vector1(angle / kk)
            else:
                self.rotate_around_perp_vector1(angle)

        elif command == 'rotate_around_perp_vector2':
            angle = float(parts[1])
            if abs(angle) > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_perp_vector2(angle / kk)
            else:
                self.rotate_around_perp_vector2(angle)
        elif command == 'rotate_around_normal_vector':
            angle = float(parts[1])
            if abs(angle) > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_normal_vector(angle / kk)
            else:
                self.rotate_around_normal_vector(angle)
        elif command == 'half_resolution':
            self.half_resolution()
        elif command == 'double_resolution':
            self.double_resolution()
        elif command == 'move_center':
            diff_x, diff_y, diff_z = float(
                parts[1]), float(parts[2]), float(parts[3])
            diff = np.array([diff_x, diff_y, diff_z]) * im_creator.center.uq
            self.move_center(diff)
        elif command == 'move_center_sim_coordinates':
            diff_x, diff_y, diff_z = float(
                parts[1]), float(parts[2]), float(parts[3])
            diff = np.array([diff_x, diff_y, diff_z]) * im_creator.center.uq
            self.move_center_sim_coordinates(diff)
        elif command == 'reset_center':
            self.reset_center()
        elif command == 'change_snapnum':
            self.change_snapnum(int(parts[1]))
        else:
            raise RuntimeError(f'unknown command in log file!\n{line}\n{command}')

        if not self.logging:
            self.frame_num += 1
        self.line_index += 1

    def _make_waypoint(self, waypoint_folder, basename, frame_num):

        image_file = ImageWriter(self.image_creator, basedir=waypoint_folder,
                                 basename=f'waypoint_{basename}_frame_{frame_num}')
        image_file.finalize()

    def expand_log(self, zoom_max=1.02, max_angle=1, verbose=False, outfolder=None, basename=None):
        self.zoom_max = zoom_max
        self.max_angle = max_angle

        self.read_log(outfolder=self.outfolder, basename=self.basename)

        if outfolder is not None:
            self.outfolder = outfolder
        if basename is not None:
            self.basename = basename

        self.basename = self.basename + '_expanded'

        self.create_log(outfolder=self.outfolder, basename=self.basename)

        for ii in range(4, len(self.lines)):
            self.step(verbose=verbose)

    def change_snapnum(self, snapnum):
        self.make_image_creator(snapnum)

        if self.logging:
            self.mylogger(f'change_snapnum,{snapnum}')

    def zoom(self, factor):
        im_creator = self.image_creator
        im_creator.zoom(factor)
        if self.logging:
            self.mylogger(f'zoom,{factor}')

    def reset_center(self):
        self.image_creator.reset_center()
        if self.logging:
            self.mylogger("reset_center")

    # def center_on_sub(self, sub_id):
    #     im_creator = self.image_creator
    #     new_center = im_creator.snap.Cat.Sub['SubhaloPos'][sub_id].T
    #     im_creator._diff_center += new_center - im_creator._center
    #     im_creator.center = new_center.copy

    # def center_on_group(self, group_id):
    #     im_creator = self.image_creator
    #     new_center = im_creator.snap.Cat.Group['GroupPos'][group_id].T
    #     im_creator._diff_center += new_center - im_creator._center
    #     im_creator.center = new_center.copy

    def width(self, arg):
        im_creator = self.image_creator
        im_creator.width = arg * im_creator.width.uq
        if self.logging:
            self.mylogger(f'width,{arg}')

    def height(self, arg):
        im_creator = self.image_creator
        im_creator.height = arg * im_creator.height.uq
        if self.logging:
            self.mylogger(f'height,{arg}')

    def depth(self, arg):
        im_creator = self.image_creator
        im_creator.depth = arg * im_creator.depth.uq
        if self.logging:
            self.mylogger(f'depth,{arg}')

    def half_resolution(self):
        im_creator = self.image_creator
        im_creator.half_resolution
        if self.logging:
            self.mylogger('half_resolution')

    def double_resolution(self):
        im_creator = self.image_creator
        im_creator.double_resolution
        if self.logging:
            self.mylogger('double_resolution')

    def rotate_around_perp_vector1(self, arg):
        im_creator = self.image_creator
        im_creator.orientation.rotate_around_perp_vector1(degrees=arg)
        if self.logging:
            self.mylogger(f'rotate_around_perp_vector1,{arg}')

    def rotate_around_perp_vector2(self, arg):
        im_creator = self.image_creator
        im_creator.orientation.rotate_around_perp_vector2(degrees=arg)
        if self.logging:
            self.mylogger(f'rotate_around_perp_vector2,{arg}')

    def rotate_around_normal_vector(self, arg):
        im_creator = self.image_creator
        im_creator.orientation.rotate_around_normal_vector(degrees=arg)
        if self.logging:
            self.mylogger(f'rotate_around_normal_vector,{arg}')

    def move_center(self, diff):
        im_creator = self.image_creator
        im_creator.move_center_along_perp_vector1(diff[0])
        im_creator.move_center_along_perp_vector2(diff[1])
        im_creator.move_center_along_normal_vector(diff[2])
        if self.logging:
            self.mylogger(f"move_center,{diff[0].value},{diff[1].value},{diff[2].value}")

    def move_center_sim_coordinates(self, diff):
        im_creator = self.image_creator
        new_center = diff + im_creator.center
        im_creator.center = new_center
        im_creator._diff_center += diff
        if self.logging:
            self.mylogger(f"move_center_sim_coordinates,{diff[0].value},{diff[1].value},{diff[2].value}")
