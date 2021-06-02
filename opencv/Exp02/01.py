import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('opencv/cascades/haarcascade_frontalface_default.xml') # 加载人脸特征库
eye_cascade = cv2.CascadeClassifier('opencv/cascades/haarcascade_eye.xml')

while(True):

    ret, frame = cap.read() # 读取一帧的图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 5, minSize = (5, 5)) # 检测人脸
    for(x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 用矩形圈出人脸
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # 释放摄像头
cv2.destroyAllWindows()