import cv2
import numpy
import os

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)
grayImage = flatNumpyArray.reshape(300, 400)
#显示转换和重构
cv2.imwrite('opencv/img/RandomGaray.png', grayImage)
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('opencv/img/RandomColor.png', bgrImage)