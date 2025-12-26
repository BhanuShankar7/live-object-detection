import cv2
import threading
import time

class CamStream:
    def __init__(self, src=0, width=1280, height=720):
        """
        Initialize the camera stream.
        
        Args:
            src: Camera source (default 0 for webcam)
            width: Desired width
            height: Desired height
        """
        # Initialize the camera stream.
        # cv2.CAP_DSHOW is recommended for Windows to enable DirectShow, 
        # often fixing initialization delays or errors.
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # Set camera resolution (might not work on all cameras, but good to try)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read the first frame to ensure it's working
        self.grabbed, self.frame = self.stream.read()
        
        # Threading checks
        self.stopped = False
        self.lock = threading.Lock()
        
        if not self.grabbed:
            print("[ERROR] Could not read from webcam. Please check your camera.")
            self.stop()

    def start(self):
        """Start the thread to read frames from the video stream."""
        if not self.stopped:
            t = threading.Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return self

    def update(self):
        """Keep looping infinitely until the thread is stopped."""
        while True:
            if self.stopped:
                self.stream.release()
                return

            grabbed, frame = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                else:
                    # If we lose signal, we might want to stop or retry
                    pass
            
            # small sleep to prevent CPU hogging if camera is slow, 
            # though usually read() blocks so this isn't strictly necessary.
            # Removing sleep as read() is blocking.

    def read(self):
        """Return the most recent frame."""
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True
