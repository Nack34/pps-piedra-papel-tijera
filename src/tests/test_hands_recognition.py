# VER https://www.youtube.com/watch?v=fd6u1TW_AGY&list=PL1FZnkj4ad1P9gulU2Ud6y-1m1fKXTPGW&index=3

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)