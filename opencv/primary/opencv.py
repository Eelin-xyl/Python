import cv2

filename = "E:\\omen.png"
img = cv2.imread(filename)
print(type(img))
print(img.shape)
print(img.dtype)