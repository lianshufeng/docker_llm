# built-in dependencies
from typing import Union

import cv2
import numpy as np
# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger
# from deepface.commons import image_utils
# 3rd party dependencies
from flask import Blueprint, request
from werkzeug.datastructures import FileStorage

logger = Logger()

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except


# 加载 OpenCV 的人脸检测分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

default_face_size = 128


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str, faceSize: int = default_face_size, keepSourceImage: bool = False) -> Union[
    str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
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

        img = imdecode(file, faceSize, keepSourceImage)
        # 判断非None
        if img is None:
            raise ValueError(f"Images doesn't have face in {img_key}")

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = (request.is_json and request.get_json()) or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()

    try:
        img = extract_image_from_request("img", int(input_args.get("face_size", default_face_size)),bool(input_args.get("keep_source_image", False)))
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])
    # actions is the only argument instance of list or tuple
    # if request is form data, input args can either be text or file
    if isinstance(actions, str):
        actions = (
            actions.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "")
            .split(",")
        )

    demographies = service.analyze(
        img_path=img,
        actions=actions,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(demographies)

    return demographies


# def load_image_from_file_storage(file: FileStorage):
#     image_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
#     return imdecode(image_array)


def imdecode(file: FileStorage, faceSize: int, keepSourceImage: bool = False):
    image_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
    # 解码图片
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 保留原始图片不需要提取人脸
    if keepSourceImage:
        return image

    # # 将图片转换为灰度图（用于人脸检测）
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 使用 Haar 分类器检测人脸
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))

    # 将图片转换为灰度图（用于人脸检测）
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Haar 分类器检测人脸
    # faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5, minSize=(32, 32))

    if len(faces) > 0:
        # 存储最大的面部
        largest_face = None
        max_area = 0

        # 遍历所有检测到的人脸
        for (x, y, w, h) in faces:
            # 计算每张人脸的面积
            area = w * h

            # 选择面积最大的（即离摄像头最近的）人脸
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)

        # 如果找到了最大的（最靠近的）人脸
        if largest_face:
            (x, y, w, h) = largest_face
            # 裁剪出最大的人脸区域
            face_region = image[y:y + h, x:x + w]

            # 压缩分辨率（可选）
            face_region = cv2.resize(face_region, (faceSize, faceSize))

            # 显示人脸
            # cv2.imshow("Face Detection", face_region)
            # cv2.waitKey(0)

            return face_region

    # 如果没有检测到任何人脸，返回 None
    return None
