
def test_arepo_image(load_catalog=False):
    import paicos as pa
    import numpy as np
    import h5py

    snap = pa.Snapshot(pa.root_dir + '/data/', 247,
                       basename='reduced_snap', load_catalog=load_catalog)

    # A length-3 array giving the center of the image.
    center = [250000, 400000, 500000]

    # This is a length-3 array giving the widths of the image.
    widths = [10000, 10000, 2000]

    # The viewing direction. Set this to e.g. 'x', 'y' or 'z'.
    direction = 'z'

    # Using the base class for image creation
    image_creator = pa.ImageCreator(snap, center, widths, direction)

    # Create arepo image file.
    # The file will have 'tmp_' prepended to the filename until .finalize()
    # is called.
    image_file = pa.ArepoImage(image_creator, pa.root_dir + 'test_data/',
                               basename='test_arepo_image_format')

    # Save some images to the file (in a real example one would first import\\
    # and use a projection function)
    image_file.save_image('Density', np.random.random((200, 200)))
    image_file.save_image('Masses', np.random.random((400, 400)))
    image_file.save_image('Velocities', np.random.random((50, 40, 3)))

    # Move from temporary filename to final filename
    image_file.finalize()

    snap.load_data(0, 'Coordinates')
    # Now amend the file with another set of data
    image_file = pa.ArepoImage(image_creator, pa.root_dir + 'test_data/',
                               basename='test_arepo_image_format', mode='a')
    data = np.random.random((500, 500))

    # Notice that we here also save attributes for coordinates,
    # these can be used to convert from comoving to non-comoving
    image_file.save_image('Coordinates', data)

    # Let us also save information about the 10 most massive FOF groups
    # (sorted according to their M200_crit)
    if load_catalog:
        index = np.argsort(snap.Cat.Group['Group_M_Crit200'])[::-1]
        for key in snap.Cat.Group.keys():
            image_file.write_data(key, snap.Cat.Group[key][index[:10]],
                                  group='Catalog',
                                  group_attrs={'Description':
                                  'Most massive FOFs'})

    image_file.finalize()

    with h5py.File(image_file.filename, 'r') as f:
        print(list(f.keys()))
        print(list(f['image_info'].keys()))
        print(dict(f['Coordinates'].attrs))
        if load_catalog:
            print(list(f['Catalog'].keys()))


if __name__ == '__main__':
    test_arepo_image(load_catalog=True)
