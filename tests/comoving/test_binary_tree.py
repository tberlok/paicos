
def test_binary_tree():
    import paicos as pa
    import numpy as np
    from scipy.spatial import KDTree
    pa.use_units(False)
    from paicos.trees.bvh_cpu import BinaryTree
    snap = pa.Snapshot(pa.root_dir + 'data/snap_247.hdf5')
    center = snap.Cat.Group['GroupPos'][0]
    widths = np.array([15000, 15000, 15000])

    # Select a cube
    index = pa.util.get_index_of_cubic_region_plus_thin_layer(
        snap['0_Coordinates'], center, widths,
        snap['0_Diameters'], snap.box)

    snap = snap.select(index, parttype=0)

    pos = snap['0_Coordinates']
    sizes = snap['0_Diameters']

    # Construct a binary tree
    bvh_tree = BinaryTree(pos, 1.2 * sizes)

    bvh_dist, bvh_ids = bvh_tree.nearest_neighbor(pos)

    # Construct a scipy kd tree
    kd_tree = KDTree(pos)

    kd_dist, kd_ids = kd_tree.query(pos)

    np.testing.assert_array_equal(kd_ids, bvh_ids)

    # Random query positions
    pos2 = np.random.rand(10000, 3) * widths[None, :]
    pos2 += (center - widths / 2)[None, :]

    bvh_dist2, bvh_ids2 = bvh_tree.nearest_neighbor(pos2)
    kd_dist2, kd_ids2 = kd_tree.query(pos2)

    np.testing.assert_array_equal(kd_ids2, bvh_ids2)
    pa.use_units(True)


if __name__ == '__main__':
    test_binary_tree()
