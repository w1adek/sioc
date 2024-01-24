import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
import filters

image = io.imread("butterfly.jpg")#, as_gray = True)

# Convolution
def convolve(image, kernel):
    conv_image = ndimage.convolve(image, kernel)
    return conv_image
    
def convolve_rgb(image, kernel):
    conv_image_dims = []
    for dim in range(3):
        conv_image_dim = ndimage.convolve(image[:, :, dim], kernel)
        conv_image_dims.append(conv_image_dim)

    conv_image = np.stack(conv_image_dims, axis=2).astype("uint8")
    return conv_image

filtered_image = convolve_rgb(image, filters.mean_filter)

fig = plt.figure(figsize=(10, 5))

fig.add_subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original") 

fig.add_subplot(1, 2, 2)
plt.imshow(filtered_image)
plt.title("Convolved") 

plt.show()