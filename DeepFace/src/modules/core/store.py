# built-in dependencies
import os
import time

import cv2
import numpy as np
# project dependencies
from deepface.commons.logger import Logger
from elasticsearch import Elasticsearch, NotFoundError
# from deepface.commons import image_utils
# 3rd party dependencies
from flask import Blueprint, request

from .routes import modelName
from .routes import represent

es_client = {}

logger = Logger()
store = Blueprint("routes/store", __name__)

# 设置最大尺寸
default_image_max_size = 640

# 创建索引
index_pre_name = "faces"


def indexName(model_name):
    return (index_pre_name + "_" + model_name).lower()


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


def create_es_client():
    # 读取环境变量
    es_hosts = os.getenv('ELASTICSEARCH_HOSTS')
    es_password = os.getenv('ELASTIC_PASSWORD')
    #  通过http的方式来连接
    client = Elasticsearch(
        [es_hosts],  # Elasticsearch服务器的URL
        http_auth=("elastic", es_password),  # 如果你的Elasticsearch有认证
    )
    return client


# 连接到 Elasticsearch
def get_es_client(model_name: str, dims: int) -> Elasticsearch:
    global es_client
    # 索引名称
    index_name = indexName(model_name)

    # 优先读取缓存
    client = es_client.get(index_name)
    if client is not None:
        return client

    # 创建es连接
    client = create_es_client()

    # 测试连接
    if not client.ping():
        client == None
        print("es 连接失败")
        return None

    print("es 连接成功")

    # 注意： 必须要创建索引
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "face_vector": {
                        "type": "dense_vector",  # 向量字段类型
                        "dims": dims,  # 向量维度
                        "index": True,  # 启用索引
                        "similarity": "cosine"  # 向量相似性算法，可选 "cosine", "l2_norm", "dot_product"
                    },
                    "key": {
                        "type": "keyword"  # 普通字段，用于存储标识符
                    },
                    "description": {
                        "type": "text"  # 文本字段，用于全文搜索
                    }
                }
            }
        })
        print("es 索引创建成功")
    es_client[index_name] = client
    return client


@store.route("/store/put", methods=["POST"])
def put():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    model_name = modelName(input_args)
    index_name = indexName(model_name)
    key = input_args.get("key", None)
    if key is None:
        return {"error": "key is not none"}, 400

    try:
        rep = represent()
        if rep[1] is not 200:
            return {"error": rep[0]}, rep[1]
        embedding = rep[0]['results'][0]['embedding']
        es_client = get_es_client(model_name, len(embedding))
        es_client.index(index=index_name, id=key, document={
            "face_vector": embedding,
            "key": key
        })

    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400

    return rep


@store.route("/store/get", methods=["POST"])
def get():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    model_name = modelName(input_args)
    index_name = indexName(model_name)
    key = input_args.get("key", None)
    if key is None:
        return {"error": "key is not none"}, 400

    try:
        response = create_es_client().get(index=index_name, id=key)
    except NotFoundError as err:
        return {'found': False}, 200
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400
    return response.body


@store.route("/store/remove", methods=["POST"])
def remove():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    model_name = modelName(input_args)
    index_name = indexName(model_name)
    key = input_args.get("key", None)
    if key is None:
        return {"error": "key is not none"}, 400
    try:
        response = create_es_client().delete(index=index_name, id=key)
    except NotFoundError as err:
        return {"result": "deleted"}, 200
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400
    return response.body


@store.route("/store/clean", methods=["POST"])
def clean():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    model_name = modelName(input_args)
    index_name = indexName(model_name)
    try:
        response = create_es_client().delete_by_query(index=index_name, body={
            "query": {
                "match_all": {}
            }
        })
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400
    return response.body


@store.route("/store/size", methods=["POST"])
def size():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    model_name = modelName(input_args)
    index_name = indexName(model_name)
    try:
        response = create_es_client().count(index=index_name, body={
            "query": {
                "match_all": {}
            }
        })
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400
    return response.body


@store.route("/store/search", methods=["POST"])
def search():
    input_args = (request.is_json and request.get_json()) or request.form.to_dict()
    max_size = input_args.get("max_size", 1)
    model_name = modelName(input_args)
    index_name = indexName(model_name)

    ret = {'time': {'represent': 0.0, 'search': 0.0}, 'items': []}
    try:
        recordTime = time.time()
        rep = represent()
        ret['time']['represent'] = float(time.time() - recordTime)
        if rep[1] is not 200:
            return {"error": rep[0]}, rep[1]
        embedding = rep[0]['results'][0]['embedding']

        es_client = get_es_client(model_name, len(embedding))

        # 查询 Elasticsearch 中的所有文档
        body = {
            "knn": {
                "field": "face_vector",
                "query_vector": embedding,
                "k": max_size,
                "num_candidates": 100
            },
            "fields": ["_id"],
            "_source": ["key"]
        }
        recordTime = time.time()
        res = es_client.search(index=index_name, size=max_size, body=body)
        ret['time']['search'] = float(time.time() - recordTime)
        for hit in res['hits']['hits']:
            ret['items'].append({
                'key': hit['_id'],
                'score': hit['_score']
            })
    except Exception as err:
        print(err)
        return {"exception": str(err)}, 400

    return ret
