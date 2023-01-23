import paicos as pa

im = pa.ImageReader(pa.root_dir + '/data/', 247,
                    basename='test_arepo_image_format')

print(im['Density'][:, :])
