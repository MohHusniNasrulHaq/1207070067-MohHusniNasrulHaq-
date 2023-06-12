import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data

image = cv2.imread("harimau.jpg",0)

image_canny = cv2.Canny (image, 100, 200)

fig, axes = plt.subplots (2,2, figsize = (20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap = 'gray')
ax[0].set_title ("Citra Input")
ax[1].hist (image.ravel(), bins = 256)
ax[1].set_title ("Histogram Citra Input")

ax[2].imshow (image_canny, cmap = 'gray')
ax[2].set_title ("Citra Output Canny")

ax[3].hist (image_canny.ravel(), bins = 256)
ax[3].set_title ("Histogram Citra Output")

plt.show()