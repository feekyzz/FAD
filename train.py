import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('C:\\Studyproject\\TVCG\\YOLOv8.2\\ultralytics\\cfg\\models\\Add\\yolov8s-AFPNHead4-FADCBlock.yaml')
    # model = YOLO('C:\\Studyproject\\TVCG\\YOLOv8.2\\ultralytics\\cfg\\models\\Add\\yolov8s-FreqFusion-2layers.yaml')
    model = YOLO('C:\\Studyproject\\TVCG\\YOLOv8.2\\ultralytics\\cfg\\models\\Add\\yolov8s-FreqFusion-4layers-2.yaml')
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov8s.yaml就是使用的v8s,
    # 类似某个改进的yaml文件名称为yolov8-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov8l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov8m.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(
        data=r"C:\Studyproject\TVCG\YOLOv8.2\kitti.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=300,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume=, # 这里是填写last.pt地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='yolov8s-FreqFusion-4layers-2',
                )

