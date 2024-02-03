import matplotlib.pyplot as plt
import scipy.fftpack as fp
from skimage.color import rgb2gray

image = plt.imread('image.png')[:, :, :3]
image = rgb2gray(image)
#image = image[:, :, :3].mean(axis=2)
image_fft = fp.fft2(image)
#image = fp.fftshift(image)
#image_fft = np.log(1+np.abs(image_fft))

image_copy = image_fft.copy()

r, c = image_copy.shape
keep_fraction = 0.1
image_copy[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
image_copy[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

image_filtered = fp.ifft2(image_copy).real

fig, axs = plt.subplots(1, 2, figsize=(6,6))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original image')
axs[1].imshow(image_filtered, cmap='gray')
axs[1].set_title('Filtered Image')
plt.show()