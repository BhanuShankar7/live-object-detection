import cv2
import time
import os

# Fix for OpenMP runtime conflict (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from camera import CamStream
from model import ObjectDetector
from utils import plot_boxes, draw_info

# Initialize helper for trackbar (opencv requires a callback)
def nothing(x):
    pass

def main():
    print("--- Starting Live Object Detection System ---")
    
    # 1. Initialize Camera
    print("[INFO] Starting Camera Stream...")
    cam = CamStream(src=0).start()
    time.sleep(1.0) # Allow camera to warm up
    
    # 2. Initialize Model
    # Switch to 'yolov8s.pt' (Small) for better accuracy
    detector = ObjectDetector(model_path='yolov8s.pt')
    
    # 3. Setup GUI
    WINDOW_NAME = 'Real-Time Object Detection'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720) # Match camera
    
    # Trackbar for confidence threshold (0 to 100, mapped to 0.0-1.0)
    # Default 45 -> 0.45 (Tuned per recommendation)
    cv2.createTrackbar('Confidence', WINDOW_NAME, 45, 100, nothing)
    
    # Variables
    detection_enabled = True
    prev_time = 0
    fps = 0
    
    print("[INFO] System Ready.")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'd' - Toggle Detection ON/OFF")
    
    # Main Loop
    while True:
        # Check if window was closed by user
        # Note: getWindowProperty returns -1 if not supported, so we check for exact 0 (closed)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
            print("[INFO] Window closed by user.")
            break

        # Read frame
        frame = cam.read()
        
        if frame is None:
            # Avoid high CPU usage if camera is not ready
            time.sleep(0.01)
            continue
        
        curr_time = time.time()
        
        # Get parameter from Trackbar safely
        try:
            conf_val = cv2.getTrackbarPos('Confidence', WINDOW_NAME)
        except cv2.error:
            # Fallback if window issue occurs
            conf_val = 50
            
        conf_threshold = conf_val / 100.0
        
        # Inference
        if detection_enabled:
            results = detector.predict(frame, conf_threshold=conf_threshold)
            
            # Post-processing & Drawing
            # detector.model.names gives us the class map
            frame = plot_boxes(frame, results, detector.model.names)
        
        # FPS Calculation
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Draw Interface (FPS, Status)
        draw_info(frame, fps, detection_enabled)
        
        # Display
        cv2.imshow(WINDOW_NAME, frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")
        elif key == ord('d'):
            detection_enabled = not detection_enabled
            state = "ON" if detection_enabled else "OFF"
            print(f"[INFO] Detection {state}")

    # Cleanup
    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] Exiting...")

if __name__ == "__main__":
    main()
