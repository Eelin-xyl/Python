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

circles = np.uitnt16(np.around(circles))
for i in circles[0, :]:

    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255),3)

cv2.imwrite("planets_circles.jpg", planets)
cv2.imshow("HoughCirlces", planets)
cv2.waitKey()
cv2.destroyAllWindows()