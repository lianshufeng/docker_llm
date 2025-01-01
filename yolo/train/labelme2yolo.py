import os
import json
import argparse
import random
import shutil
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

def extract_class_mapping(input_dir):
    """自动从输入目录中提取类标标签映射"""
    class_mapping = {}
    class_id = 0

    # 遍历所有JSON文件
    for json_file in Path(input_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for shape in data['shapes']:
            label = shape['label']
            if label not in class_mapping:
                class_mapping[label] = class_id
                class_id += 1

    return class_mapping

def convert_labelme_to_yolo(json_file, output_dir, class_mapping, img_dir, label_dir):
    """将LabelMe格式转换为YOLO格式"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取图像文件路径和尺寸
    img_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    # 获取YOLO格式的标注文件路径
    base_filename = Path(json_file).stem
    output_txt = Path(label_dir) / (base_filename + '.txt')

    with open(output_txt, 'w') as out_file:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # 假设是矩形标注，获取左上角和右下角坐标
            x_min = min([point[0] for point in points])
            y_min = min([point[1] for point in points])
            x_max = max([point[0] for point in points])
            y_max = max([point[1] for point in points])

            # 获取YOLO类标ID
            class_id = class_mapping.get(label, None)
            if class_id is None:
                continue

            # 转换为YOLO格式 (class_id, center_x, center_y, width, height)，并归一化
            center_x = (x_min + x_max) / 2 / img_width
            center_y = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # 写入YOLO格式标注信息
            out_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

    # 复制图像到目标文件夹
    shutil.copy(img_path, img_dir)

def create_data_split(input_dir, output_dir, class_mapping):
    """创建数据集并按比例划分训练集和验证集"""
    # 设置输出目录结构
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_train_dir = images_dir / 'train'
    images_val_dir = images_dir / 'val'
    labels_train_dir = labels_dir / 'train'
    labels_val_dir = labels_dir / 'val'

    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有的JSON文件路径
    json_files = list(Path(input_dir).glob("*.json"))
    random.shuffle(json_files)

    # 划分数据集（80%训练集，20%验证集）
    train_files = json_files[:int(0.8 * len(json_files))]
    val_files = json_files[int(0.8 * len(json_files)):]

    # 使用 tqdm 显示进度条，转换训练集
    print("Converting training set...")
    for json_file in tqdm(train_files, desc="Training set", unit="file"):
        convert_labelme_to_yolo(json_file, output_dir, class_mapping, images_train_dir, labels_train_dir)

    # 使用 tqdm 显示进度条，转换验证集
    print("Converting validation set...")
    for json_file in tqdm(val_files, desc="Validation set", unit="file"):
        convert_labelme_to_yolo(json_file, output_dir, class_mapping, images_val_dir, labels_val_dir)

    # 创建训练集和验证集的图片路径列表（相对路径）
    train_img_paths = [str(Path('images/train') / img.name) for img in images_train_dir.glob("*")]
    val_img_paths = [str(Path('images/val') / img.name) for img in images_val_dir.glob("*")]

    # 保存训练集和验证集的图片路径到txt文件
    with open(Path(output_dir) / 'train.txt', 'w') as f:
        f.write("\n".join(train_img_paths) + "\n")

    with open(Path(output_dir) / 'val.txt', 'w') as f:
        f.write("\n".join(val_img_paths) + "\n")

    # 创建YAML配置文件
    create_yaml(output_dir)

    print(f"Dataset has been split. Training set: {len(train_files)} images, Validation set: {len(val_files)} images.")

def create_yaml(output_dir):
    """生成YOLO所需的yaml配置文件"""
    yaml_data = {
        'train': './images/train',
        'val': './images/val',
        'nc': len(class_mapping),  # 类别数
        'names': list(class_mapping.keys())  # 类别名称
    }
    
    # 保存YAML文件
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    print(f"YAML configuration file created at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to YOLO format and split into train/val.")
    parser.add_argument("input", type=str, help="Input directory containing LabelMe JSON files.")
    parser.add_argument("output", type=str, help="Output directory to save YOLO format labels and images.")

    args = parser.parse_args()

    # 自动提取class_mapping
    class_mapping = extract_class_mapping(args.input)

    # 创建数据集并划分为训练集和验证集
    create_data_split(args.input, args.output, class_mapping)

    # 输出类别映射信息
    print("Class Mapping:", class_mapping)
