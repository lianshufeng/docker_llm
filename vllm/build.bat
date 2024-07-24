@echo off
docker build -f Dockerfile-cuda12.1 -t lianshufeng/llm:vllm-cuda12.1 ./
pause