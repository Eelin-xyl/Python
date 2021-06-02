import sys,os,cv2
import os
import numpy as np

# 加载面部资料
def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: 人面数据所在的文件路径
        sz: 图片尺寸设置

    Returns:
        A list [X,y]
            X: 图片信息
            y: 图片读取数据
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
                # os.path.join进行路径的拼接 eg: D:\auto\dc
            print(subject_path)
            for filename in os.listdir(subject_path):
                # os.listdir(subject_path)是一个列表 
                print(filename)
                if (filename == ".directory"):
                    continue
                filepath = os.path.join(subject_path, filename)
                print(filepath)
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # 灰度图
                # resize to given size (if given)
                if (sz is not None):
                    im = cv2.resize(im, sz) # 图片缩放
                X.append(np.asarray(im, dtype=np.uint8))
                print(X)
                y.append(c)
                print(y)
            c = c+1
    return [X, y]
    
# 面部识别
def face_rec():
    # names = ['bjx','dc', 'dk','ds','lh','aals111','wls','zww']
    [X, y] = read_images('D:\\auto\\')
    # y = [0, 0, 0, 0, 1, 1, 1, 1]


    # 创建识别模型，使用EigenFace算法识别，Confidence评分低于4000是可靠
    # model = cv2.face.EigenFaceRecognizer_create()
    # 创建识别模型，使用LBPHFace算法识别，Confidence评分低于50是可靠
    model = cv2.face.LBPHFaceRecognizer_create()
    # 创建识别模型，使用FisherFace算法识别，Confidence评分低于4000是可靠
    # model = cv2.face.FisherFaceRecognizer_create()

    # 训练模型
    # train函数参数：images, labels，两参数必须为np.array格式，而且labels的值必须为整型
    model.train(np.array(X), np.array(y))

    # 开启摄像头
    camera = cv2.VideoCapture(0)
    # 加载Haar级联数据文件，用于检测人面
    face_cascade = cv2.CascadeClassifier('D:/python/opencv/cascades/haarcascade_frontalface_default.xml')
    eye_path = 'D:/python/opencv/cascades/haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(eye_path)

    while (True):
        # 检测摄像头的人面
        read, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # 将检测的人面进行识别处理
        for (x, y, w, h) in faces:
            # 画出人面所在位置并灰度处理
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = img[y:(y + h), x:(x + w)]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imshow("",roi_gray)
            
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # 将检测的人面缩放200*200大小，用于识别
            # cv2.INTER_LINEAR是图片变换方式，其余变换方式如下：
            # INTER_NN - 最近邻插值。
            # INTER_LINEAR - 双线性插值(缺省使用)
            # INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。
            # INTER_CUBIC - 立方插值。
            try:
                roi = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)

                # 检测的人面与模型进行匹配识别
                # predict（）函数做比对，返回一个元祖格式值 （标签，系数）。系数和算法有关，
                params = model.predict(roi)
                print(params)
                print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                # 人面识别也是通过摄像头来获取人物头像，然后根据算法与文件夹dc和dk的头像进行匹配对比。
                # 匹配成功会返回Label和Confidence，其中Label代表资料加载顺序，即函数read_images读取文件夹dc和dk的读取顺序。
                # Confidence是识别评分，每种算法有一定的范围值，符合范围值才算匹配成功。
                path="D:/auto"  #待读取的文件夹
                path_list=os.listdir(path)
                path_list.sort()  #对读取的路径进行排序
                # print(len(path_list))
                for i in range(0, len(path_list)):
                    if params[0] == i:
                        name = path_list[i]
                # 将识别结果显示在摄像头上
                # cv2.FONT_HERSHEY_SIMPLEX 定义字体
                # cv2.putText参数含义：图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                cv2.putText(img, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            except:
                continue
        cv2.imshow("camera",img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    face_rec()