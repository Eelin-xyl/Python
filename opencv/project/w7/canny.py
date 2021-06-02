import cv2
import numpy as np

#读入灰度图像
img = cv2.imread("opencv/project/w7/statue_small.jpg", cv2.IMREAD_GRAYSCALE)
#边缘检测
cv2.imwrite("opencv/project/w7/canny.jpg", cv2.Canny(img, 200, 300))
#显示图像
cv2.imshow('canny', cv2.imread("opencv/project/w7/canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()