
def test_gpu_binary_tree():
    import cupy as cp
    from numba import cuda
    import paicos as pa
    from scipy.spatial import KDTree
    from paicos.trees.bvh_gpu import GpuBinaryTree
    import numpy as np
    pa.use_units(False)
    snap = pa.Snapshot(pa.root_dir + 'data/snap_247.hdf5')
    center = snap.Cat.Group['GroupPos'][0]
    widths = np.array([15000, 15000, 15000])

    index = pa.util.get_index_of_cubic_region_plus_thin_layer(
        snap['0_Coordinates'], center, widths,
        2.0 * snap['0_Diameters'], snap.box)

    snap = snap.select(index, parttype=0)

    pos_cpu = snap['0_Coordinates']
    pos = cp.array(pos_cpu)
    sizes = cp.array(snap['0_Diameters'])
    bvh_tree = GpuBinaryTree(pos, 1.2 * sizes)

    bvh_dist, bvh_ids = bvh_tree.nearest_neighbor(pos)

    # Construct a scipy kd tree
    kd_tree = KDTree(pos_cpu)

    kd_dist, kd_ids = kd_tree.query(pos_cpu)

    np.testing.assert_array_equal(kd_ids, bvh_ids.get())

    # Random query positions
    pos2_cpu = np.random.rand(10000, 3) * widths[None, :]
    pos2_cpu += (center - widths / 2)[None, :]

    pos2 = cp.array(pos2_cpu)

    bvh_dist2, bvh_ids2 = bvh_tree.nearest_neighbor(pos2)
    kd_dist2, kd_ids2 = kd_tree.query(pos2_cpu)

    np.testing.assert_array_equal(kd_ids2, bvh_ids2.get())
    pa.use_units(True)


if __name__ == '__main__':
    test_gpu_binary_tree()
