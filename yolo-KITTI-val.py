from ultralytics import YOLO

if __name__ == '__main__':

    model = (YOLO("runs/v8m-FEM3/weights/best.pt"))

    model.val(split='val')
    # model.val(split='val', save_conf=True, save_txt=True)