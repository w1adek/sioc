import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn import metrics

"""Interpolation kernels"""
def h1(x):
    return np.where((x >= -0.5) & (x < 0.5), 1, 0)
def h2(x):
    return np.where((x >= -1) & (x <= 1), 1-np.abs(x), 0)

original_image = io.imread('image_scaling/pic.jpg')
image_gray = rgb2gray(original_image)

scale_factor = 2
downsize_kernel = np.ones([scale_factor, scale_factor]) / np.size(np.ones([scale_factor, scale_factor]))

"""Downsizing"""
def convolution(image, kernel, step):
    image_width = image.shape[1]
    image_height = image.shape[0]
    downsized_image = np.zeros([image_height // step, image_width // step])
    
    for y in range(0, image_height, step):
        for x in range(0, image_width, step):
            #new_pixel = (image[y : y+step, x : x+step]).max() # Max Pooling algorithm for downsampling
            new_pixel = (image[y : y+step, x : x+step] * kernel).sum()
            downsized_image[y // step, x // step] = new_pixel
            
    return downsized_image

"""Upsizing"""
def upsizing(image, scale_factor, kernel):
    image_width = image.shape[1]
    image_height = image.shape[0]
    long_image = np.zeros([image_height, image_width * scale_factor])
    
    x_axis = np.linspace(0, 1, image_width)
    x_interp = np.linspace(0, 1, image_width * scale_factor)
    
    for y in range(image_height):
        current_row = image[y, :]
        x_shifted = (x_interp - x_axis[:, np.newaxis]) / np.diff(x_axis)[0]
        interp_row = np.dot(current_row, kernel(x_shifted))
        long_image[y, :] = interp_row
    
    image_width = long_image.shape[1]
    image_height = long_image.shape[0]
    upsized_image = np.zeros([image_height * scale_factor, image_width])
    
    x_axis = np.linspace(0, 1, image_height)
    x_interp = np.linspace(0, 1, image_height * scale_factor)
    
    for x in range(image_width):
        current_column = long_image[:, x]
        x_shifted = (x_interp - x_axis[:, np.newaxis]) / np.diff(x_axis)[0]
        interp_column = np.dot(current_column, kernel(x_shifted))
        
        upsized_image[:, x] = interp_column
    
    return upsized_image

"""MSE"""
def mse_compare(original_image, upsized_image):
    original_image = original_image.reshape(-1)
    upsized_image = upsized_image.reshape(-1)
    mse = metrics.mean_squared_error(original_image, upsized_image)
    return f'MSE: {mse:.4f}'

downsized_image = convolution(image_gray, downsize_kernel, scale_factor)
upsized_image = upsizing(downsized_image, scale_factor, h2)
print(mse_compare(image_gray, upsized_image))

io.imshow(upsized_image)
plt.show()