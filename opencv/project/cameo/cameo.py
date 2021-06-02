import cv2
from managers import WindowManager, CaptureManager

class Cameo(object):

    def __init__(self):

        self._windowManager = WindowManager("Cameo", self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
    
    def run(self):

        """Run the main loop."""

        self._windowManager.createWindow()

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                # TODO: rilter the frame (Chaper 3).
                pass
            
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    def onKeypress(self, keycode):

        """Handle a keypress.

        space   ->   Take a screenshot.
        tab     ->   Start/Stop recording a screencast.
        escape  ->   Quit.

        """

        if keycode == 32: # space
            self._captureManager.writeImage('cameo/img/screenshot.png')
        
        elif keycode == 9: # tab          
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('cameo/img/screencast.avi')
            
            else:
                self._captureManager.stopWritingVideo()
        
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    Cameo().run()