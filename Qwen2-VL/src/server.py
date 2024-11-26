import threading
from argparse import ArgumentParser

from flask import Flask, request, jsonify, send_from_directory
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




@app.route('/compatible-mode/v1', methods=['POST'])
@cross_origin()
def compatibleModeV1():
    global model, processor, args
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        # print(messages)

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
        print(output_text)
        return jsonify(output_text), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    global args
    args = _get_args()

    # 启动一个线程载入模型
    # threading.Thread(target=load_model).start()
    load_model()
    app.run(debug=False, port=args.server_port, host=args.server_name)
