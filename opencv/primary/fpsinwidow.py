import cv2

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow') #指定窗口名
cv2.setMouseCallback('MyWindow', onMouse) #获取鼠标输入
print("Showing camera feed. Click window or press any key to stop")
success, frame = cameraCapture.read()
while success and not clicked and cv2.waitKey(1) == -1: #没达到停止条件时
    cv2.imshow("MyWindow", frame)
    success, frame = cameraCapture.read()

cv2.destroyAllWindows()
cameraCapture.release()