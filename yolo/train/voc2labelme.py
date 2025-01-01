import os
import json
import xml.etree.ElementTree as ET
import cv2
import sys
from tqdm import tqdm

def find_voc_dirs(input_voc_dir):
    """
    自动探测 VOC 数据集的相关目录 (Annotations 和 JPEGImages)
    
    Args:
        input_voc_dir (str): 数据集根目录
    
    Returns:
        tuple: 发现的 `Annotations` 目录和 `JPEGImages` 目录路径
    """
    annotations_dir = None
    images_dir = None
    
    # 遍历根目录，寻找可能的 Annotations 和 JPEGImages 目录
    for root, dirs, files in os.walk(input_voc_dir):
        # 如果发现包含 XML 文件的目录，认为是 Annotations 目录
        if not annotations_dir and any(f.endswith('.xml') for f in files):
            annotations_dir = root
        
        # 如果发现包含图像文件的目录，认为是 JPEGImages 目录
        if not images_dir and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            images_dir = root
        
        # 如果两个目录都已找到，则可以提前退出
        if annotations_dir and images_dir:
            break
    
    if not annotations_dir or not images_dir:
        raise FileNotFoundError("Could not find both 'Annotations' and 'JPEGImages' directories.")
    
    return annotations_dir, images_dir

def voc_to_labelme_inplace(input_voc_dir):
    """
    将 Pascal VOC 格式数据集转换为 Labelme JSON 格式。
    JSON 文件直接存储在图像所在的目录中。

    Args:
        input_voc_dir (str): VOC 数据集根目录（包含 `Annotations` 和 `JPEGImages`）
    """
    # 自动探测 VOC 数据集的相关目录
    annotations_dir, images_dir = find_voc_dirs(input_voc_dir)
    
    # 获取所有的 XML 文件路径
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    # 使用 tqdm 显示进度条
    for xml_file in tqdm(xml_files, desc="Processing VOC to Labelme", unit="file"):
        xml_path = os.path.join(annotations_dir, xml_file)
        img_file = os.path.splitext(xml_file)[0] + '.jpg'
        img_path = os.path.join(images_dir, img_file)

        # 检查对应图像是否存在
        if not os.path.exists(img_path):
            continue

        try:
            # 解析 VOC XML 文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像信息
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image file could not be loaded: {img_path}")
            height, width, _ = img.shape

            # 构造 Labelme 格式
            labelme_data = {
                "version": "4.5.9",  # Labelme 版本
                "flags": {},
                "shapes": [],
                "imagePath": img_file,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }

            # 添加标注信息
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                shape = {
                    "label": label,
                    "points": [[xmin, ymin], [xmax, ymax]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)

            # 保存为 JSON 文件
            json_filename = os.path.splitext(img_file)[0] + ".json"
            json_path = os.path.join(images_dir, json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(labelme_data, json_file, indent=4)

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("Usage: python voc2labelme.py <input_voc_dir>")
        sys.exit(1)

    # 获取输入目录
    input_voc_dir = sys.argv[1]

    # 执行转换
    try:
        voc_to_labelme_inplace(input_voc_dir)
    except Exception as e:
        print(f"Error: {e}")
