# Offline Real-Time Object Detection System

A complete real-time object detection application using Python, OpenCV, and YOLOv8. Designed to run offline without external APIs.

## Features
- **Real-time Detection**: Uses threaded camera capture for specific FPS performance.
- **Offline Capabilities**: Runs entirely locally using Ultralytics YOLOv8.
- **Interactive UI**:
    - Toggle detection On/Off ('d')
    - Adjustable Confidence Threshold Slider
    - Screenshot capture ('s')
    - FPS Counter
- **Modular Design**: Separated concerns (Camera, Model, Utils) for easy maintenance.

## Requirements
- Python 3.8+
- Webcam

## Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: The first run will automatically download the `yolov8n.pt` model (approx 6MB) from Ultralytics. After that, it works completely offline.*

## Usage

Run the main script:

```bash
python main.py
```

### Controls
| Key | Action |
| :--- | :--- |
| `q` | **Quit** the application |
| `s` | **Save** the current frame to disk |
| `d` | **Toggle** detection (ON/OFF) |
| **Slider** | Adjust **Confidence Threshold** |

## Customization

### Replacing the Model
To use a different model (e.g., a custom trained one or `yolov8s.pt` for higher accuracy):
1.  Place your `.pt` file in the project folder.
2.  Edit `main.py`:
    ```python
    detector = ObjectDetector(model_path='your_custom_model.pt')
    ```

### Retraining
You can retrain YOLOv8 on your own dataset (COCO format) easily:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='coco128.yaml', epochs=100)
```

## Structure
- `main.py`: Main entry point and UI loop.
- `camera.py`: Threaded camera stream handler.
- `model.py`: YOLOv8 wrapper for inference.
- `utils.py`: Drawing and helper functions.
"# live-object-detection" 
