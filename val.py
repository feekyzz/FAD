import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs\\train\\yolov8s-FADC2\\weights\\best.pt')
    model.val(data=r'C:\\Studyproject\\TVCG\\YOLOv8.2\\kitti.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='Freq2+AFPN4',
              )