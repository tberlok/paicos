import pytest

def test_writer_and_reader():

    import paicos as pa
    import numpy as np

    snap = pa.Snapshot(pa.data_dir, 247, basename='reduced_snap', load_catalog=False)

    writer = pa.PaicosWriter(snap, basedir=pa.data_dir)

    writer.write_data('0_Density', snap['0_Density'][0])

    writer.write_data('nested_test_data', snap['0_Density'][0], group='group1/group2/group3')

    writer.finalize()

    f = pa.PaicosReader(writer.filename)


    # Amend mode
    writer = pa.PaicosWriter(snap, basedir=pa.data_dir, mode='a')

    writer.write_data('0_Density2', snap['0_Density'][0])
    writer.write_data('test_data', snap['0_Density'][0], group='group1')

    # Read-plus mode
    writer = pa.PaicosWriter(snap, basedir=pa.data_dir, mode='r+')
    writer.write_data('0_Density2', 2 * snap['0_Density'][0])

    f = pa.PaicosReader(writer.filename)

    np.testing.assert_array_equal(2 * f['0_Density'], f['0_Density2'])


if __name__ == '__main__':
    test_slicer_writer_and_reader(True)
