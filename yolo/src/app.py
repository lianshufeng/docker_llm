from argparse import ArgumentParser

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from ultralytics import YOLO

# 初始化Flask应用
app = Flask(__name__)


def load_model():
    global model, args
    # 加载YOLOv10模型
    model = YOLO(args.model)

    # args.device 不能为 None
    # if args.device != None:
    #     model.to(args.device)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='./models.pt',
                        help='Checkpoint name or path, default to %(default)r')
    # parser.add_argument('--device',
    #                     type=str,
    #                     default=None,
    #                     help='cuda or cpu')
    parser.add_argument('--server-port', type=int, default=5000, help='server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='server name.')
    return parser.parse_args()


# 定义API接口来处理流式数据
@app.route('/detect', methods=['POST'])
@cross_origin()
def detect():
    # 获取图片流
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image file provided'}), 400

    # 将图片流转换为OpenCV格式
    img_bytes = file.read()

    # 将图片保存到磁盘上,文件名取格林威治时间
    # with open('c:/output/'+str(int(round(time.time() * 1000)))+'.jpg', 'wb') as f:
    #     f.write(img_bytes)
    #
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 开始推理
    results = model.predict(source=img, save=False)[0]

    # 获取结果
    labels = results.names
    detected_objects = []

    # 处理结果
    for result in results:
        box = result.boxes
        xyxy = box.xyxy
        conf = box.conf
        cls = box.cls

        # 返回结果
        ret = {
            'label': labels[int(cls)],
            'confidence': conf.item(),
            'bounding_box': [int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])]
        }

        # 获取关键点
        if result.keypoints != None:
            keypoints = []
            for kpt in result.keypoints.cpu().numpy()[0]:
                for xy in kpt.xy[0]:
                    keypoints.append([int(xy[0]), int(xy[1])])
            ret['keypoints'] = keypoints

        detected_objects.append(ret)

    # 返回检测结果
    return jsonify({'detected_objects': detected_objects})


if __name__ == '__main__':
    global args
    args = _get_args()

    load_model()

    app.run(debug=False, port=args.server_port, host=args.server_name, threaded=True)
