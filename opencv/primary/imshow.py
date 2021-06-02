import cv2
import numpy as numpy

img = cv2.imread('opencv/img/1.png')
img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('opencv/img/1.1.png', img_new)
cv2.imshow("My IMG", img)
cv2.imshow("Gray", img_new)
cv2.waitKey()
cv2.destroyWindow('My IMG')