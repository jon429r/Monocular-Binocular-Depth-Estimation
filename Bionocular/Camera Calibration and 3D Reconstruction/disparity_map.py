import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('assets/left01.jpg', 0)
imgR = cv2.imread('assets/right01.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()
