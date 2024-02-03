import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fp
from skimage.color import rgb2gray

image = plt.imread('test.jpg')

# Grayscale input image
image = rgb2gray(image)

# Fourier transform
image_fft = fp.fft2(image)

# Sort frequencies
sorted_freq = np.sort(np.abs(image_fft.flatten()))

# Remove the low 99% of frequencies
remove = 0.99 * len(sorted_freq)
threshold_freq = sorted_freq[int(remove)]

# Boolean mask
threshold_image = np.abs(image_fft) > threshold_freq

# Apply mask to fft image
trunc_image = image_fft * threshold_image

# Compression ratio
"""
out1 = np.count_nonzero(image_fft)
out2 = np.count_nonzero(trunc_image)
print(out1/out2)
"""

# Inverse Fourier transform
compressed_image = fp.ifft2(trunc_image).real

fig, axs = plt.subplots(1, 2, figsize=(6,6))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(compressed_image, cmap='gray')
axs[1].set_title('Compressed image')
plt.show()