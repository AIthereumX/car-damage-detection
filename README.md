# car-damage-detection
combine YOLO and EfficientNet
gathering yolo dataset from the link blow
https://universe.roboflow.com/car-damage-detection-wjnwh/car-damage-detection-krsix

# YOLO + EfficientNet Vehicle Damage Detection & Classification

A hybrid pipeline for **automated vehicle damage detection** using YOLO (for object detection) plus EfficientNet (for severity/type classification on cropped regions).  
This project is suitable for academic/industrial applications and can be quickly run or customized.

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [YOLO Training](#yolo-training)
- [Damage Cropping](#damage-cropping)
- [EfficientNet Training](#efficientnet-training)
- [Hybrid Pipeline Usage](#hybrid-pipeline-usage)
- [Example Inference Script](#example-inference-script)
- [Future Work / TODOs](#future-work--todos)

---

## Installation

Make sure you have Python 3.8+ with pip.

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas efficientnet_pytorch
```

---

## Dataset Preparation

- The dataset must contain vehicle images and **YOLO-style labels** (for each image: a `.txt` file with bounding box info).
- Example folder structure:

```
/datasets
  /images
    train/
    val/
  /labels
    train/
    val/
```

- For severity/type classification:  
  You’ll also need cropped damage images with class folders (e.g. "minor", "moderate", "severe").

---

## YOLO Training

Train a YOLO model for **damage localization**:

```bash
yolo task=detect mode=train data=your_data.yaml model=yolov8s.pt epochs=50 imgsz=640
```
- Edit `your_data.yaml` with your dataset paths.

---

## Damage Region Cropping

Detect and crop damaged regions from test images using trained YOLO:

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
results = model("test.jpg")

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    crop = cv2.imread("test.jpg")[y1:y2, x1:x2]
    cv2.imwrite(f"crop_{i}.jpg", crop)
```

---

## EfficientNet Training

Train on cropped regions for **damage severity/type**:

- Arrange damage crops in class folders:
  ```
  /damage_crops
    /minor/
    /moderate/
    /severe/
  ```
- Use any EfficientNet implementation (keras or pytorch). Sample with [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch):

```python
from efficientnet_pytorch import EfficientNet
# Standard training loop with PyTorch image datasets, etc.
```

---

## Hybrid Pipeline Usage

**Full workflow:**
1. Input an image
2. YOLO detects and draws bounding boxes for damages
3. Each box is cropped
4. EfficientNet classifies the severity/type for each crop
5. Output: localized & classified damages

---

## Example Inference Script

```python
from ultralytics import YOLO
import cv2
from efficientnet_pytorch import EfficientNet

image = "car.jpg"
yolo_model = YOLO("best.pt")
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0') # or your trained weights

results = yolo_model(image)
for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    crop = cv2.imread(image)[y1:y2, x1:x2]
    # Your custom code: preprocess, then run EfficientNet...
    # pred = efficientnet_model(preprocessed_crop)
    # severity = decode(pred)
    print(f"Box {i}: (coords: {x1},{y1},{x2},{y2}) - Severity: [predict here]")
```

---

## Future Work / TODOs

- Integrate **segmentation** for more precise damage masks
- Try other classifiers (ViT, ResNet, etc.)
- Web service / REST API demo
- Iranian car dataset integration
- Ensemble, post-processing, severity reasoning

---

**Pull requests, issues and suggestions are very welcome!**  
*Made with :heart: for automotive AI research.*

---
