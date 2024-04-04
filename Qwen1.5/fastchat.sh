#/!bin/bash

nohup python3 -m fastchat.serve.controller --host 0.0.0.0 &
nohup python3 -m fastchat.serve.model_worker --model-names "gpt-3.5-turbo,Qwen1.5" --model-path /models --device cpu --host 0.0.0.0 &
nohup python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000

