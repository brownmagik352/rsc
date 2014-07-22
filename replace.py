#!/usr/bin/python
import numpy as np
import cv2



big = cv2.imread('cup1.jpg',-1)
img = cv2.resize(big, (0,0), fx=0.2, fy=0.2) 

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_bound = np.array([165,50,50])
upper_bound = np.array([179,255,255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
res = cv2.bitwise_and(img,img, mask= mask)


cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()

# # BGR -> RGB with matplotlib
# from matplotlib import pyplot as plt
# plt.subplot(231),plt.imshow(img),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(mask),plt.title('MASK')
# plt.subplot(233),plt.imshow(res),plt.title('RESULT')

# plt.show()