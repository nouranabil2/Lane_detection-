import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test1.jpg")
img1 =  np.copy(img)
GRID_SIZE = 20
height,width ,channel = img.shape
# get grid image to extract the pts coordinates 
for x in range(0, width -1, GRID_SIZE):
     cv2.line(img, (x, 0), (x, height), (255, 0, 0), 1, 1)
key = cv2.waitKey(0) 
plt.imshow(img)