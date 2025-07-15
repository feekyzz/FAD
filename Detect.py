import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:\\Studyproject\\TVCG\\YOLOv8.2\\runs\\train\\yolov8s-FADC2\\weights\\best.pt') # select your model.pt path
    model.predict(source=r'C:\Studyproject\TVCG\YOLOv8.2\val_imgs',
                  imgsz=640,
                  project='runs/detect',
                  name='test',
                  save=True,
                  save_txt=True
                  # classes=0, 是否指定检测某个类别.
                )