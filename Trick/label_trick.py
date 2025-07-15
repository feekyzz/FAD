import os

# 原始文件夹和新文件夹路径
#best
# original_path = r"C:\Studyproject\TVCG\YOLOv8.2\runs\detect\predict2\labels"
# new_path =  r"C:\Studyproject\TVCG\YOLOv8.2\runs\detect\predict2\2_trick"
#bad
original_path = r"C:\Studyproject\TVCG\YOLOv8.2\runs\detect\Bad89.3\labels"
new_path =  r"C:\Studyproject\TVCG\YOLOv8.2\runs\detect\Bad89.3\2_trick"

# 确保新目录存在
os.makedirs(new_path, exist_ok=True)

# 遍历所有txt文件
for file_name in os.listdir(original_path):
    if file_name.endswith(".txt"):
        original_file = os.path.join(original_path, file_name)
        new_file = os.path.join(new_path, file_name)

        # 读取文件并处理内容
        with open(original_file, "r") as f:
            lines = f.readlines()

        # 处理每一行，去掉最后一个数值
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                new_lines.append(" ".join(parts[:-1]))  # 去掉最后一个数值

        # 保存到新路径
        with open(new_file, "w") as f:
            f.write("\n".join(new_lines) + "\n")

print("处理完成，新的文件已保存到:", new_path)
