import base64
import os
import tempfile
import uuid
from argparse import ArgumentParser
from io import BytesIO

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

app = Flask(__name__)


def load_model():
    global model, processor, args
    model_path = args.model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='/models',
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='cuda or cpu')

    parser.add_argument('--min-pixels',
                        type=int,
                        default=256 * 28 * 28,
                        help='set min_pixels')

    parser.add_argument('--max-pixels',
                        type=int,
                        default=1280 * 28 * 28,
                        help='set min_pixels')
    parser.add_argument('--max-new-tokens',
                        type=int,
                        default=128,
                        help='set min_pixels')

    parser.add_argument('--server-port', type=int, default=8000, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Demo server name.')

    return parser.parse_args()


def itemToLocalFile(image, files):
    # 循环 items;
    if image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
            image_rgb = image_obj.convert('RGB')
            # 生成一个随机的文件名
            file = tempfile.gettempdir() + '/' + str(uuid.uuid4()) + ".jpg"
            # 图片保存到本地
            image_rgb.save(file)
            image_obj.close()
            image_rgb.close()
            files.append(file)
            return "file://" + file
    return None


def toLocalFile(content, name):
    files = []
    if not name in content:
        return files
    image = content[name]
    if isinstance(image, list):
        # 循环数组，需要下标
        for index, item in enumerate(image):
            ret = itemToLocalFile(item, files)
            if ret is not None:
                content[name][index] = ret
    else:
        content[name] = itemToLocalFile(image, files)
    return files
    # if image.startswith("data:image"):
    #     data = image.split(";", 1)[1]
    #     if data.startswith("base64,"):
    #         data = base64.b64decode(data[7:])
    #         image_obj = Image.open(BytesIO(data))
    #         # 生成一个随机的文件名
    #         file = tempfile.gettempdir() + '/' + str(uuid.uuid4()) + ".jpg"
    #         # 图片保存到本地
    #         image_obj.save(file)
    #         files.append(file)
    #         image_obj.close()
    #         # 修改content
    #         content[name] = "file://" + file
    # return files


# 转换为本地的
def convertLocalFile(messages):
    files = []
    # 循环content
    for message in messages:
        for content in message['content']:
            # 添加集合
            files += toLocalFile(content, 'image')
            files += toLocalFile(content, 'video')
    return files


@app.route('/compatible-mode/v1', methods=['POST'])
@cross_origin()
def compatibleModeV1():
    global model, processor, args
    files = []
    try:
        data = request.get_json()
        messages = data.get('messages', [])

        # base64 转 本地文件
        files += convertLocalFile(messages)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Preparation for inference
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device=args.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 定义一个dic
        output_text = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": output_text[0],
                        "role": "assistant",
                        "function_call": None,
                        "tool_calls": None
                    }
                }
            ]
        }

        print(output_text)
        return jsonify(output_text), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    finally:
        # 删除 files 的所有文件
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    global args
    args = _get_args()

    # 启动一个线程载入模型
    # threading.Thread(target=load_model).start()
    load_model()
    app.run(debug=False, port=args.server_port, host=args.server_name)
