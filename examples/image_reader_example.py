import paicos as pa

im = pa.ImageReader(pa.data_dir + 'test_data/', 247,
                    basename='projection_x')

print(im['0_Density'][:, :])
print(im.extent)
print(im.center)
print(im.direction)
print(im.image_creator)
