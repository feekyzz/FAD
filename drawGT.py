import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--imgs', type=str, default=r'C:\Studyproject\TVCG\YOLOv8.2\val_imgs', help='数据目录')
parser.add_argument('--labels', type=str, default=r'C:\Studyproject\ITS\yolov8\val_label', help='labels目录')
parser.add_argument('--save', type=str, default='save', help='保存目录')
args = parser.parse_args()

# 输入图片文件夹
img_folder = args.imgs
img_list = os.listdir(img_folder)
img_list.sort()
# 输入标签文件夹
label_folder = args.labels
label_list = os.listdir(label_folder)
label_list.sort()
# 输出图片文件夹位置
output_folder = args.save

# 类别标签（与标签文件中的类别名称对应）
labels = ['Car', 'Pedestrian', 'Cyclist']  # 确保顺序与标签文件中的类别一致

# 颜色映射（BGR格式）
colormap = [(255, 255, 255), (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255)]


def xywh2xyxy(x, w1, h1, img):
    """将归一化的xywh坐标转换为xyxy像素坐标，并绘制边界框和标签"""
    label, x, y, w, h = x

    # 确保类别索引是整数
    try:
        label_ind = labels.index(label)  # 根据类别名称获取索引
    except ValueError:
        print(f"Warning: Unknown label '{label}'. Skipping...")
        return img

    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    p1 = (int(top_left_x), int(top_left_y))
    p2 = (int(bottom_right_x), int(bottom_right_y))

    # 绘制矩形框
    cv2.rectangle(img, p1, p2, colormap[label_ind + 1], thickness=2, lineType=cv2.LINE_AA)

    # 绘制标签
    label_text = labels[label_ind]
    if label_text:
        text_size = cv2.getTextSize(label_text, 0, fontScale=2/3, thickness=2)[0]
        outside = p1[1] - text_size[1] - 3 >= 0  # 标签是否适合框外
        p2_text = (p1[0] + text_size[0], p1[1] - text_size[1] - 3 if outside else p1[1] + text_size[1] + 3)
        cv2.rectangle(img, p1, p2_text, colormap[label_ind + 1], -1, cv2.LINE_AA)  # 填充标签背景
        cv2.putText(
            img,
            label_text,
            (p1[0], p1[1] - 2 if outside else p1[1] + text_size[1] + 2),
            0,
            2/3,
            colormap[0],  # 文字颜色（白色）
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return img


if __name__ == '__main__':
    # 创建输出文件夹
    if Path(output_folder).exists():
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(img_list):
        if not img_name.endswith('.jpg'):
            continue

        # 读取图像
        image_path = os.path.join(img_folder, img_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping...")
            continue

        h, w = img.shape[:2]

        # 读取标签文件
        label_path = os.path.join(label_folder, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found. Skipping...")
            continue

        with open(label_path, 'r') as f:
            lines = f.read().strip().splitlines()
            lb = []
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid line format in {label_path}: '{line}'. Skipping...")
                    continue
                label = parts[0]  # 类别名称（字符串）
                coords = list(map(float, parts[1:]))  # 坐标（浮点数）
                lb.append([label] + coords)  # 合并为 [label, x, y, w, h]

        # 绘制每一个目标
        for x in lb:
            img = xywh2xyxy(x, w, h, img)

        # 保存结果
        output_path = os.path.join(output_folder, f"{Path(img_name).stem}.png")
        cv2.imwrite(output_path, img)