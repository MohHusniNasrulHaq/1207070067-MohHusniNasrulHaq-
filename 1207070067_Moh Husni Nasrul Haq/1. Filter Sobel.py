import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data

image = cv2.imread("harimau.jpg",0)

image_sobelx = cv2.Sobel(image,cv2.CV_8U, 1, 0, ksize = 5)
image_sobely = cv2.Sobel(image,cv2.CV_8U, 0, 1, ksize = 5)
image_sobel = image_sobelx + image_sobely

fig, axes = plt.subplots (4,2, figsize = (20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap = 'gray')
ax[0].set_title ("Citra Input")
ax[1].hist (image.ravel(), bins = 256)
ax[1].set_title ("Histogram Citra Input")

ax[2].imshow (image_sobelx, cmap = 'gray')
ax[2].set_title ("Citra Output")

ax[3].hist (image_sobelx.ravel(), bins = 256)
ax[3].set_title ("Histogram Citra Input")

ax[4].imshow (image_sobely, cmap = 'gray')
ax[4].set_title ("Citra Output")

ax[5].hist (image_sobely.ravel(), bins = 256)
ax[5].set_title ("Histogram Citra Input")

ax[6].imshow (image_sobel, cmap = 'gray')
ax[6].set_title ("Citra Output")

ax[7].hist (image_sobel.ravel(), bins = 256)
ax[7].set_title ("Histogram Citra Input")

fig.tight_layout()
plt.show()