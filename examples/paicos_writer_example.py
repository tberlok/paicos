import paicos as pa

snap = pa.Snapshot(pa.data_dir, 247, basename='reduced_snap', load_catalog=False)

writer = pa.PaicosWriter(snap, basedir=pa.data_dir)

writer.write_data('0_Density', snap['0_Density'][0])

writer.write_data('nested_test_data', snap['0_Density'][0], group='group1/group2/group3')

writer.finalize()

f = pa.PaicosReader(writer.filename)

print(f)
