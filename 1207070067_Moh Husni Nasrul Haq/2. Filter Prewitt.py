import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data

image = cv2.imread("harimau.jpg",0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

image_prewittx = cv2.filter2D(image, -1, kernelx)
image_prewitty = cv2.filter2D(image, -1, kernely)
image_prewitt = image_prewittx + image_prewitty

fig, axes = plt.subplots (4,2, figsize = (20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap = 'gray')
ax[0].set_title ("Citra Input")
ax[1].hist (image.ravel(), bins = 256)
ax[1].set_title ("Histogram Citra Input")

ax[2].imshow (image_prewittx, cmap = 'gray')
ax[2].set_title ("Citra Output Prewitt X")

ax[3].hist (image_prewittx.ravel(), bins = 256)
ax[3].set_title ("Histogram Citra Output Prewitt X")

ax[4].imshow (image_prewitty, cmap = 'gray')
ax[4].set_title ("Citra Output Prewitt Y")

ax[5].hist (image_prewitty.ravel(), bins = 256)
ax[5].set_title ("Histogram Citra Output Prewitt Y")

ax[6].imshow (image_prewitt, cmap = 'gray')
ax[6].set_title ("Citra Output Prewitt")

ax[7].hist (image_prewitt.ravel(), bins = 256)
ax[7].set_title ("Histogram Citra Output Prewitt")

fig.tight_layout()
plt.show()