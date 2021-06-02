import cv2
import numpy as np
#读入图像
img = cv2.imread('opencv/w8/lines.jpg')
#转换颜色空间
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#边缘检测
edges = cv2.Canny(gray, 50, 120)
#最小直线长度
minLineLength = 100
#最大线段间隙
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[1]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('edges', edges)
cv2.imshow('lines', img)
cv2.waitKey()
cv2.destroyAllWindows()