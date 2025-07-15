import os
from PIL import Image

# 转换YOLO输出坐标到像素坐标

# 类别映射
class_map = {
    0: "Car",
    1: "Cyclist",
    2: "Pedestrian"
}

# 图片目录和预测结果文件目录
image_folder = "datasets/KITTI_100/images/val"  # 修改为你的图片文件夹路径
predictions_folder = "runs/detect/predict/labels"  # 修改为预测结果文件夹路径
output_folder = "runs/detect/predict/test"  # 修改为你希望保存新txt文件的目录

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 读取预测结果并转换为图片坐标
def process_predictions(prediction_file, image_file):
    # 读取图片的实际尺寸
    image_path = os.path.join(image_folder, image_file)
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    with open(prediction_file, 'r') as file:
        lines = file.readlines()

    results = []
    for line in lines:
        # 解析每行内容
        parts = line.split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5])

        # 使用实际的图片尺寸进行坐标计算
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)

        # 获取类别名称
        class_name = class_map.get(class_id, "Unknown")

        # 格式化输出：类别 置信度 x_min y_min x_max y_max
        results.append(f"{class_name} {confidence:.4f} {x_min} {y_min} {x_max} {y_max}")

    return results


# 处理所有图片的预测，并为每张图片生成对应的txt文件
def process_all_images():
    for image_file in os.listdir(image_folder):
        # 获取对应的预测文件
        image_name, _ = os.path.splitext(image_file)
        prediction_file = os.path.join(predictions_folder, f"{image_name}.txt")

        if os.path.exists(prediction_file):
            results = process_predictions(prediction_file, image_file)

            # 将结果写入新的输出目录（与图片同名的txt文件）
            output_file = os.path.join(output_folder, f"{image_name}.txt")
            with open(output_file, 'w') as out_file:
                for result in results:
                    out_file.write(result + '\n')


if __name__ == "__main__":
    process_all_images()
    print(f"处理完成，所有图片的预测结果已生成并保存在新目录：{output_folder}")