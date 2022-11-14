from cyoctree import CyOctree
import numpy as np

np.random.seed(42)

num_points = 10**6

# Create random points for initializing a tree
pos = np.random.rand(num_points, 3)
tree = CyOctree(pos, n_ref=8)
print('created tree')

search_radius = tree.node_sizes.min()


def nearest_neighbor(point):
    r = np.sqrt(np.sum((pos - point)**2, axis=1))
    return np.argmin(r)


points = np.random.rand(10, 3)
for ii in range(10):
    # print(ii)
    point = points[ii, :]
    index = tree.nearest_neighbor(point, radius=search_radius)
    index2 = nearest_neighbor(point)
    assert index == index2


# np.testing.assert_array_equal(np.sort(ind), np.sort(indices))
# print(ind)
# print(pos[ind, :])

# %timeit index = tree.nearest_neighbor(point, radius=search_radius)
# %nearest_neighbor(point)
