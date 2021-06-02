import numpy
import cv2

img = cv2.imread("../opencv/img/1.png")
kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
cv2.filter2D(img, 1, kernel)