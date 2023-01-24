import paicos as pa

im = pa.ImageReader(pa.root_dir + '/data/', 247,
                    basename='projection_x')

print(im['0_Density'][:, :])
print(im.extent)
print(im.center)
print(im.direction)
print(im.image_creator)
