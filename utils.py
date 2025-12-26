import cv2
import time

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 0, 0), (128, 128, 0), 
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)
]

def draw_info(frame, fps, mode_status):
    """
    Draw FPS and Status on the frame.
    """
    height, width = frame.shape[:2]
    
    # Draw FPS
    cv2.putText(
        frame, 
        f"FPS: {int(fps)}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Draw Status (Detection ON/OFF)
    status_color = (0, 255, 0) if mode_status else (0, 0, 255)
    status_text = "Detection: ON" if mode_status else "Detection: OFF (Paused)"
    cv2.putText(
        frame, 
        status_text, 
        (20, height - 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        status_color, 
        2
    )

def plot_boxes(frame, results, class_names):
    """
    Custom bounding box drawer.
    
    Args:
        frame: The original image/frame.
        results: Ultralytics YOLO results object.
        class_names: Dictionary of class names {0: 'person', ...}
    
    Returns:
        Frame with boxes drawn.
    """
    
    # Iterate through results
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Confidence
            conf = float(box.conf[0])
            
            # Class ID
            cls = int(box.cls[0])
            label = class_names.get(cls, str(cls))
            
            # Select color based on class id
            color = COLORS[cls % len(COLORS)]
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return frame
