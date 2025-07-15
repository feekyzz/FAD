此文件夹用于评估KITTI数据集分类指标

第一步，使用YOLO预测，保存预测打框图片和坐标txt文件 yolo-KITTI-predict.py

第二步，YOLO输出坐标为比例，首先要将其转换为像素坐标 1_zuobiao.py

第三步，使用2_eval.py 按照类别分别评估。输出结果手动保存


from ultralytics import YOLO
yolo = YOLO("runs/yolov10s-87.7/weights/best.pt",task="detect")
result = yolo(source=r'D:\a6.yolov10\yolov10\datasets\val_imgs\val_imgs',save=True,save_conf = True,save_txt = True)



FEM3测试结果

2D Detection AP for Car

Easy: 98.685475
Moderate: 98.803188
Hard: 90.612062

2D Detection AP for Pedestrian

Easy: 89.817966
Moderate: 87.904022
Hard: 80.001922

2D Detection AP for Cyclist

Easy: 90.657192
Moderate: 89.991410
Hard: 88.995026



YOLOv10-87.7测试结果

2D Detection AP for Car

Easy: 98.463501
Moderate: 98.649732
Hard: 90.372711

2D Detection AP for Pedestrian

Easy: 88.536029
Moderate: 79.555004
Hard: 71.245412

2D Detection AP for Cyclist

Easy: 90.312831
Moderate: 89.240931
Hard: 88.594911
