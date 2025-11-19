import numpy as np
data = np.load("data/coverage_grid.npz")
grid = data["grid"]
x_min = data["x_min"]
y_min = data["y_min"]
cell = data["cell_size"]
print(grid.shape, x_min, y_min, cell)
