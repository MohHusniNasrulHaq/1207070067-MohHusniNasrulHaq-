import cv2 
from matplotlib import pyplot as plt
 
img = cv2.imread('harimau.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img0 = cv2.GaussianBlur(gray,(3,3),0)
laplacian = cv2.Laplacian(img0,cv2.CV_64F)

plt.subplot(1,2,1) 
plt.imshow(img0, cmap='gray') 
plt.title('Original') 
plt.xticks([]) 
plt.yticks([]) 
plt.subplot(1,2,2) 
plt.imshow(laplacian, cmap='gray') 
plt.title('Laplacian') 
plt.xticks([])
plt.yticks([])
plt.show()