# Gloved vs Ungloved Hand Detection

## 📌 Overview
This project implements an object detection pipeline to identify whether a person is wearing gloves or not. The system detects two classes:

- Gloves
- Non-Gloves

It processes input images, performs detection using a trained model, and outputs annotated images along with structured JSON logs.

---

## 📂 Dataset

- **Name:** Gloves Detection Dataset  
- **Source:** Roboflow Universe  
- **Format:** YOLOv8 (PyTorch)

The dataset contains labeled images with bounding boxes for:
- Gloved hands
- Bare (non-gloved) hands

The dataset was split into:
- Training set
- Validation set

---

## 🤖 Model Used

- **Model:** YOLOv8n (Nano version)
- **Framework:** Ultralytics YOLOv8
- **Reason for selection:**
  - Lightweight and fast
  - Suitable for real-time applications
  - Easy to fine-tune on custom datasets

---

## ⚙️ Training & Preprocessing

### Training Environment:
- Google Colab (T4 GPU)

### Steps performed:
1. Uploaded dataset (ZIP) to Colab
2. Extracted dataset and fixed `data.yaml` paths
3. Installed Ultralytics YOLOv8
4. Fine-tuned pretrained model (`yolov8n.pt`) on custom dataset

### Training Parameters:
- Epochs: 20
- Image size: 640

### Preprocessing / Augmentation:
- Default YOLO augmentations were used:
  - Horizontal flipping
  - Scaling
  - Color adjustments

---

## 📊 Output

The model generates:
- Annotated images with bounding boxes
- JSON logs per image in the format:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "Gloves",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

# ✅ What Worked Well
 - YOLOv8 provided fast and reliable detection performance

# ⚠️ What Didn’t Work
  - Limited dataset size (~150 images) affected generalization
  - Model struggled in:
    - Low lighting conditions
    - partially visible hands
# How To Run
  - python detection_script.py
