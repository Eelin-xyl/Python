import cv2
def detect():
    face_path = 'D:/python/opencv/cascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_path = 'D:/python/opencv/cascades/haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_path)
        # detect()函数首先会加载Haar级联文件，由此OpenCV可执行人脸检测.
    
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 转化成灰度图很有必要，因为人脸检测需要基于灰度的色彩空间
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            # 捕获帧，对具有灰度色彩空间的帧调用detectMultiScale函数
            # 进行人脸识别

        for (x, y, w, h) in faces:
            print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 在整张图中画出人脸 左上角 右下角 线宽 
            
            roi = frame[y:(y + h), x:(x + w)]
                # ROI 感兴趣区域！在人脸矩形框创建一个相应的感兴趣区域，并在该区域中进行"眼睛检测"\
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imshow("",roi_gray)
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
                # 进行人脸中的眼睛识别
            print(eyes)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break
            
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()            