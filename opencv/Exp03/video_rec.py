import sys, os, cv2
import numpy as np

def read_images(path, sz=None):

    c = 0
    X,y = [], []

    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:

            subject_path = os.path.join(dirname, subdirname)
            print(subject_path)

            for filename in os.listdir(subject_path):

                print(filename)

                if (filename == ".directory"):
                    continue

                filepath = os.path.join(subject_path, filename)
                print(filepath)
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) 

                if (sz is not None):
                    im = cv2.resize(im, sz) 

                X.append(np.asarray(im, dtype=np.uint8))
                print(X)
                y.append(c)
                print(y)
            c = c+1

    return [X, y]
    
def face_rec():

    [X, y] = read_images('opencv/Exp03/data/')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.array(X), np.array(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('opencv/cascades/haarcascade_frontalface_default.xml')

    while (True):

        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = img[y:(y + h), x:(x + w)]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            try:
                roi = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print(params)
                print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                path="opencv/Exp03/data"
                path_list=os.listdir(path)
                path_list.sort()

                for i in range(0, len(path_list)):

                    if params[0] == i:
                        name = path_list[i]

                cv2.putText(img, name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            except:
                continue

        cv2.imshow("camera",img)
        
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    face_rec()