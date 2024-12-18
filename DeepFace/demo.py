import os
import time

from deepface import DeepFace

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet"
]

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yolov11s',
    'yolov11n',
    'yolov11m',
    'yunet',
    'centerface',
]
metrics = ["cosine", "euclidean", "euclidean_l2"]
alignment_modes = [True, False]

model_index = 0
backends_index =0

# 设置环境变量，指定使用 PyTorch 作为后端
os.environ["DEEPFACE_BACKEND"] = "pytorch"

# face verification
result = DeepFace.verify(
    model_name=models[model_index],
    img1_path="dataset/img2.jpg",
    img2_path="dataset/img5.jpg",
    detector_backend=backends[backends_index],
    align=alignment_modes[0],
)
print(result)





# embeddings
embeddings_time = time.time()
embedding_objs = DeepFace.represent(
    model_name=models[model_index],
    img_path="dataset/img2.jpg",
    detector_backend=backends[backends_index],
    align=alignment_modes[0],
)
print(embedding_objs)
print("embedding time: ", time.time() - embeddings_time)

result = DeepFace.find(
    model_name=models[model_index],
    db_path="dataset",
    img_path="dataset/img2.jpg",
    detector_backend=backends[backends_index],
    distance_metric=metrics[0],
    align=alignment_modes[0],
)
print(result)

# 进行人脸分析，使用 GPU
# result = DeepFace.analyze(
#     detector_backend=backends[0],
#     img_path="dataset/img2.jpg",
#     actions=['age', 'gender', 'race', 'emotion'],
#     enforce_detection=False
# )
# print(result)
