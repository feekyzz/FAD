from ultralytics import YOLO

# Load a model
model = (YOLO("ultralytics/cfg/models/best/yolov8s-FAD.yaml"))
# model = (YOLO("ultralytics/cfg/models/v8/yolov8s.yaml"))

# Train the moder
model.train(data='yolo-KITTI-data.yaml', workers=0, epochs=300, batch=16, name="YOLOv8s-20250714",device=0,amp=False)
