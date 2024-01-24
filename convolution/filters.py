import numpy as np

# Filters
# Edge detection
laplace_filter = np.array(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]]
)

sobelX_filter = np.array(
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]]
)

sobelY_filter = sobelX_filter.transpose()

# Blur
gaussian_filter = 1/16 * np.array(
    [[1, 2, 1],
     [1, 4, 1],
     [1, 2, 1]]
)

mean_filter = 1/9 * np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]
)

# Sharpening
shaper_filter = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]]
)