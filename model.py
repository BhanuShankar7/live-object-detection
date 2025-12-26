from ultralytics import YOLO
import torch
import cv2
import os

class ObjectDetector:
    def __init__(self, model_path='yolov8s.pt'):
        """
        Initialize the Object Detection model.
        
        Args:
            model_path: Path to the .pt model file. 
                        Defaults to 'yolov8s.pt' (Small) for better accuracy than Nano.
        """
        print(f"[INFO] Loading model from {model_path}...")
        
        # Check if CUDA is available, else use CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise e
            
        # Warmup
        import numpy as np
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_frame, device=self.device, verbose=False)
        print("[INFO] Model loaded and warmed up.")

    def predict(self, frame, conf_threshold=0.5, iou_threshold=0.45):
        """
        Perform inference/tracking on a single frame.
        
        Args:
            frame: Numpy array (image)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            
        Returns:
            results: List of detections
        """
        # Run inference with tracking (persist=True keeps IDs across frames)
        # Switching to 'track' mode significantly stabilizes detections vs 'predict'
        results = self.model.track(
            source=frame, 
            conf=conf_threshold, 
            iou=iou_threshold,
            device=self.device, 
            persist=True,      # Enable tracking memory
            verbose=False,
            imgsz=640,
            tracker="bytetrack.yaml" # Built-in lightweight tracker
        )
        return results
