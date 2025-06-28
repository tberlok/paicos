import paicos as pa

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

print(f)
