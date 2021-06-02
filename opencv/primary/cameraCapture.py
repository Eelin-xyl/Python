import cv2

cameraCapture = cv2.VideoCapture(0)
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = 30
videoWriter = cv2.VideoWriter('opencv/img/MyCameraVideo.avi', cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)
success, frame = cameraCapture.read()
numFrameRemaining = 10 * fps - 1
while success and numFrameRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFrameRemaining -= 1
cameraCapture.release()