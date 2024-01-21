
def test_radial_selection():
    import numpy as np
    import paicos as pa
    pa.use_units(True)

    snap = pa.Snapshot(pa.data_dir, 247, basename='reduced_snap',
                       load_catalog=False)
    center = [398968.4, 211682.6, 629969.9] * snap.length

    r_max = 300 * center.unit_quantity

    # Calculate index in the "slow" but simple way
    r = np.sqrt(np.sum((snap['0_Coordinates'] - center[None, :])**2., axis=1))
    index_slow = r < r_max

    # Use OpenMP parallel Cython code
    index = pa.util.get_index_of_radial_range(snap['0_Coordinates'],
                                              center, 0., r_max)

    # Check that we get the same result
    np.testing.assert_array_equal(index_slow, index)

    selected_snap = snap.select(index, parttype=0)

    selected_snap.save_new_snapshot('very_small_snap')

    r = np.sqrt(np.sum((selected_snap['0_Coordinates'] - center[None, :])**2.,
                       axis=1))

    assert np.sum(r < r_max) == r.shape[0]


if __name__ == '__main__':
    test_radial_selection()
