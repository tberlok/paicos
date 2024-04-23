import numpy as np


def find_angle_subdivisions(angle, max_angle):
    kk = 1
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
    def __init__(self, snapnum=None, image_creator=None):
        self.first_call = True
        if snapnum is None:
            assert image_creator is not None
            self.image_creator = image_creator
        else:
            self.make_image_creator(snapnum)
        self.first_call = False

        self.logging = False
        self.zoom_max = 10000
        self.max_angle = 360

    def make_image_creator(self, snapnum):
        """

        TODO: Add example implementation in the docstring of
        of this method.
        """
        err_msg = ("You need to implement this yourself.\n"
                   + "Please see the docstring of this method")
        raise RuntimeError(err_msg)

    def mylogger(self, string, line_ending='\n', mode='a'):
        outfolder = self.outfolder
        basename = self.basename
        with open(f'{outfolder}/{basename}.log', mode) as f:
            f.write(string + line_ending)

    def create_log(self, outfolder='.', basename='image'):
        self.outfolder = outfolder
        self.basename = basename
        self.mylogger('# Paicos image log file', mode='w')
        self.mylogger(outfolder)
        self.mylogger(basename)
        im_creator = self.image_creator
        image_file = pa.ArepoImage(im_creator, basedir=outfolder,
                                   basename=f'{basename}_logfile_start')

        zeros = np.zeros((im_creator.npix_width, im_creator.npix_height))
        uq = im_creator.snap.uq('')
        image_file.save_image('Zeros', zeros * uq)

        # Move from temporary filename to final filename
        image_file.finalize()
        self.mylogger(image_file.filename)
        self.logging = True

    def read_log(self, filename=None, outfolder='.', basename='image'):
        if filename is None:
            filename = f'{outfolder}/{basename}.log'
        with open(filename, 'r') as f:
            lines = f.readlines()
        outfolder = lines[1].strip()
        basename = lines[2].strip()
        self.logfile_start = lines[3].strip()
        self.lines = lines
        self.frame_num = 0
        self.line_index = 4

    def step(self):
        line = self.lines[self.line_index].strip()
        parts = line.split(',')
        command = parts[0]

        im_creator = self.image_creator

        # print(line, command)
        if command == 'center_on_sub' or command == 'center_on_group':
            cx, cy, cz = float(parts[2]), float(parts[3]), float(parts[4])
            new_center = np.array([cx, cy, cz]) * im_creator.center.uq
            raise RuntimeError("not implemented")
            # self.reset_center(new_center)
        elif command == 'zoom':
            zoom_fac = float(parts[1])
            if zoom_fac > self.zoom_max or (1 / zoom_fac) > self.zoom_max:
                kk = find_zoom_sub_divisions(zoom_fac, self.zoom_max)
                new_zoom_fac = zoom_fac**(1 / kk)
                for _ in range(kk):
                    self.zoom(new_zoom_fac)
        elif command == 'width':
            self.width(float(parts[1]))
        elif command == 'height':
            self.height(float(parts[1]))
        elif command == 'depth':
            self.depth(float(parts[1]))
        elif command == 'rotate_around_perp_vector1':
            angle = float(parts[1])
            if angle > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_perp_vector1(angle / kk)

        elif command == 'rotate_around_perp_vector2':
            angle = float(parts[1])
            if angle > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_perp_vector2(angle / kk)
        elif command == 'rotate_around_normal_vector':
            angle = float(parts[1])
            if angle > self.max_angle:
                kk = find_angle_subdivisions(angle, self.max_angle)
                for _ in range(kk):
                    self.rotate_around_normal_vector(angle / kk)
        elif command == 'half_resolution':
            self.half_resolution()
        elif command == 'double_resolution':
            self.double_resolution()
        elif command == 'move_center':
            diff_x, diff_y, diff_z = float(parts[1]), float(parts[2]), float(parts[3])
            diff = np.array([diff_x, diff_y, diff_z]) * im_creator.center.uq
            self.move_center(diff)
        elif command == 'move_center_sim_coordinates':
            diff_x, diff_y, diff_z = float(parts[1]), float(parts[2]), float(parts[3])
            diff = np.array([diff_x, diff_y, diff_z]) * im_creator.center.uq
            self.move_center_sim_coordinates(diff)
        elif command == 'reset_center':
            self.reset_center()
        else:
            raise RuntimeError(f'unknown command in log file!\n{line}\n{command}')

        self.frame_num += 1
        self.line_index += 1

    def expand_log(self, outfolder='.', basename='image', zoom_max=1.02, max_angle=1):
        self.zoom_max = zoom_max
        self.max_angle = max_angle
        self.outfolder = outfolder
        self.basename = basename + '_expanded'
        self.create_log(outfolder=self.outfolder, basename=self.basename)
        self.read_log(outfolder=self.outfolder, basename=basename)

        for ii in range(4, len(self.lines)):
            self.step()

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


if __name__ == '__main__':
    import paicos as pa

    class Actions2(Actions):
        def make_image_creator(self, snapnum):
            snap = pa.Snapshot(pa.data_dir, snapnum)

            if self.first_call:
                center = snap.Cat.Group['GroupPos'][0]
                widths = [15000., 15000., 15000.]
                orientation = pa.Orientation(normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
            else:
                orientation = self.image_creator.orientation.copy
                widths = np.copy(self.image_creator.widths)
                center = np.copy(self.image_creator.center)
                del self.image_creator
            self.image_creator = pa.ImageCreator(snap, center, widths, orientation)

    actions = Actions2(247)
    actions.create_log(outfolder='.', basename='image')
    actions.rotate_around_normal_vector(10)
    actions.zoom(10.)
    actions.change_snapnum(500)
    actions.rotate_around_normal_vector(10)
    actions.rotate_around_perp_vector1(-20)
    actions.rotate_around_perp_vector2(15)
    actions.move_center(actions.image_creator.center * 0.05)
    actions.move_center_sim_coordinates(-actions.image_creator.center * 0.05)
    actions.zoom(10.)
    actions.reset_center()

    # Now, let's expand the log file
    actions = Actions(247)
    actions.read_log(outfolder='.', basename='image')
    # TODO: read_log should also set the image creator
    info = pa.ImageReader(actions.logfile_start)
    actions.expand_log(outfolder='.', basename='image')

# if __name__ == '__main__':
#     import paicos as pa
#     snap = pa.Snapshot(pa.data_dir + 'snap_247.hdf5')
#     center = snap.Cat.Group['GroupPos'][0]
#     R200c = snap.Cat.Group['Group_R_Crit200'][0].value
#     widths = [15000., 15000., 15000.]
#     orientation = pa.Orientation(
#         normal_vector=[0, 0, 1], perp_vector1=[1, 0, 0])
#     im = pa.ImageCreator(snap, center, widths, orientation)

#     actions = Actions(im)
#     actions.create_log(outfolder='.', basename='image')
# actions.rotate_around_normal_vector(10)
# actions.rotate_around_perp_vector1(-20)
# actions.rotate_around_perp_vector2(15)
# actions.move_center(center * 0.05)
# actions.move_center_sim_coordinates(-center * 0.05)
# actions.zoom(10.)
# actions.reset_center()

#     # Now, let's expand the log file
#     actions = Actions(im)
#     actions.read_log(outfolder='.', basename='image')
#     info = pa.ImageReader(actions.logfile_start)
#     im = pa.ImageCreator(snap, info.center, info.widths, info.orientation)
#     actions.expand_log(outfolder='.', basename='image')
