# built-in dependencies

import cv2
import numpy as np
# project dependencies
from deepface.api.src.modules.core import service
from deepface.commons.image_utils import load_image
from deepface.commons.logger import Logger
# from deepface.commons import image_utils
# 3rd party dependencies
from flask import Blueprint, request

es_client = {}

logger = Logger()
blueprint2 = Blueprint("routes/v2", __name__)


# 设置最大尺寸
default_image_max_size = 640



# @blueprint2.route("/")
# def home():
#     return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str, image_max_size: int) -> [np.ndarray, float]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
            :param img_key:
            :param image_max_size:
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        # 读取到内存
        image_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
        img, scale = imageCode(cv2.imdecode(image_array, cv2.IMREAD_COLOR), image_max_size)
        # 判断非None
        if img is None:
            raise ValueError(f"Images doesn't have face in {img_key}")

        return img, scale
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = (request.is_json and request.get_json()) or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        buf, _ = load_image(input_args.get(img_key))
        img, scale = imageCode(buf, image_max_size)
        if img is None:
            raise ValueError(f"Images doesn't have face in {img_key}")

        return img, scale

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


def modelName(input_args):
    return input_args.get("model_name", "ArcFace")


@blueprint2.route("/v2/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()

    image_max_size = int(input_args.get("image_max_size", default_image_max_size))
    detector_backend = input_args.get("detector_backend", "yunet")  # opencv
    model_name = modelName(input_args)  # VGG-Face

    try:
        img, scale = extract_image_from_request("img", image_max_size)
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=bool(input_args.get("enforce_detection", True)),
        align=bool(input_args.get("align", True)),
        anti_spoofing=bool(input_args.get("anti_spoofing", False)),
        max_faces=int(input_args.get("max_faces", 1)),
    )

    logger.debug(obj)

    # 判断obj 不为数组
    if type(obj) is not tuple:
        obj['scale'] = scale
        obj['detector_backend'] = detector_backend
        obj['model_name'] = model_name

    return obj


def imageCode(image: np.ndarray, image_max_size: int):
    #  去噪：应用高斯模糊
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # 获取图像的原始宽度和高度
    height, width = image.shape[:2]

    # 如果小于实际分辨率则直接返回
    if (height <= image_max_size and width <= image_max_size):
        return image, 1

    # 判断宽度和高度，确保最大尺寸为640
    if width > height:  # 如果宽度较大
        scale = image_max_size / width
    else:  # 如果高度较大
        scale = image_max_size / height

    # 根据缩放比例计算新的宽度和高度
    new_width = int(width * scale)
    new_height = int(height * scale)

    image = cv2.resize(image, (new_width, new_height))
    # cv2.imshow("Face Detection", image)
    # cv2.waitKey(0)

    return image, scale


