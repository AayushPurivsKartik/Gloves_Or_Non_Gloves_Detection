from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # fast model

model.train(
    data="/content/Gloves_New_Dataset/data.yaml",
    epochs=20,
    imgsz=640
)