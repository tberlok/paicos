from cyoctree import CyOctree
import numpy as np

np.random.seed(42)

num_points = 10**7
radius = 0.01

# Create random points for initializing a tree
pos = np.random.rand(num_points, 3)
tree = CyOctree(pos, n_ref=64)
print('created tree')
indices = tree.range_search(pos[0, :], radius)


def range_search(pos, radius):
    r = np.sqrt(np.sum((pos - pos[0, :])**2, axis=1))
    ind = np.where(r < radius)
    return ind[0]


ind = range_search(pos, radius)

np.testing.assert_array_equal(np.sort(ind), np.sort(indices))
# print(ind)
# print(pos[ind, :])

# %timeit tree.range_search(pos[0, :], radius)
# %timeit range_search(pos, radius)
