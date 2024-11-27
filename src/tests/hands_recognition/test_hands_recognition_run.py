from ultralytics import YOLO

base_rute = "C:/Users/NICOLAS/Desktop/2024/Proyectito con Igna/Yolov11Proyect/"
train_number = "3" # 2 = 10 epocas, 3 = 50 epocas

rute = base_rute+"runs/pose/train"+train_number+"/weights/best.pt"

model = YOLO(rute) 
results = model.predict(source=0, show=True, imgsz=640)

