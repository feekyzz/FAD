from ultralytics import YOLO

# yolo = YOLO(r"C:\Studyproject\TVCG\YOLOv8.2\runs\train\yolov8s-FADC2\weights\best.pt",task="detect")
#
# result = yolo(source=r'val_imgs',save=True,save_conf = True,save_txt = True)


yolo = YOLO(r"C:\Studyproject\TVCG\YOLOv8.2\runs\train\yolov8s-FADC\weights\best.pt",task="detect")

result = yolo(source=r'val_imgs',save=True,save_conf = True)