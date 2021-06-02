import cv2
def detect(filename):
    path = 'D:/python/opencv/cascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(path)
        # 使用OpenCV自带的haarcascade文件里现有的分类器用于检测物体
        # 其中：haarcascade_frontalface_alt.xml与haarcascade_frontalface_alt2.xml都是人脸识别的Haar特征分类器了。
        # 这行代码：加载cv2中的级联分类器haarcascade_frontalface_default

    img = cv2.imread(filename)
        # 读入图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将图片转化为灰度图(人脸检测需要)
    # face_cascade.load(path)
    faces = face_cascade.detectMultiScale(gray,1.3, 5)
        # face_cascade.detectMultiScale进行实际的人脸检测
        # detectMultiScale函数。
            # 作用：它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用
                # 参数1：image--待检测图片，一般为灰度图像加快检测速度；
                # 参数2：objects--被检测物体的矩形框向量组；
                # 参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
                # 参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个),如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除,如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框,这种设定值一般用在用户自定义对检测结果的组合程序上；
                # 参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING,如果设置为CV_HAAR_DO_CANNY_PRUNING,那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域,因此这些区域通常不会是人脸所在区域;
                # 参数6、7：minSize和maxSize用来限制得到的目标区域的范围
    for (x, y, w, h) in faces:
        img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 检测操作的返回值为人脸矩形数组。函数cv2.rectangle允许通过坐标来绘制矩形(x和y表示左上角的坐标，w和h表示人年矩形的宽度和高度)
            # 通过依次提取faces变量中的值来找到人脸，并在人脸周围绘制蓝色矩形，这是在原始图像而并不是在图像的灰色版本上进行绘制。
        print(img.shape)
        print(img2.shape)
    cv2.namedWindow("DC")
    cv2.imshow("DC", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    filename = 'D:/python/OpenCV_pactures_video/lena.jpg'
    detect(filename)