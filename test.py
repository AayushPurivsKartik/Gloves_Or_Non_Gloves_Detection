import os
import cv2
import json
from ultralytics import YOLO

model = YOLO("best.pt")

input_folder = "input"
output_folder = "output"
log_folder = "logs"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    results = model(img)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls_id]

        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imwrite(os.path.join(output_folder, img_name), img)

    json_data = {
        "filename": img_name,
        "detections": detections
    }

    with open(os.path.join(log_folder, img_name.replace(".jpg",".json")), "w") as f:
        json.dump(json_data, f, indent=2)