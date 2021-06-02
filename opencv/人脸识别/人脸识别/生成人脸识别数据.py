"""
图像是灰度格式,后缀名为.pgm
图像形状为正方形
图像大小要一样(这里使用的大小为200*200;大多数免费图像集都比这个小)

过程： 检测人脸 裁剪灰度帧区域,将其大小调整为200*200的像素 保存到指定文件
"""
import cv2
def generate():
    face_path = 'D:/python/opencv/cascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_path)
    eye_path = 'D:/python/opencv/cascades/haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_path)

    camera = cv2.VideoCapture(0)
    count = 0
    while camera.isOpened():
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                # 按照指定的宽度、高度缩放图片 详见图像几何变换
            cv2.imshow("", f)
            if count <= 50:
                cv2.imwrite("D:\\auto\\dc\\%s.pgm" % str(count), f)
                # img2 = cv2.imread("D:/python/opencv/人脸识别/人脸识别/data/%s.pgm")
                # cv2.imshow("111",img2)
            count += 1
            print(count)  
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate()
