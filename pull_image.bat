@echo off
docker pull lianshufeng/docker-pull
REM docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v %CD%:/work lianshufeng/docker-pull -proxy=http://192.168.31.98:1080 -cleanImage=true lianshufeng/llm:unsloth lianshufeng/llm:vllm-cuda12.1 lianshufeng/llm:LLaMA-Factory lianshufeng/llm:fastchat-cpu

docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v %CD%:/work lianshufeng/docker-pull -m=docker.jpy.wang -cleanImage=true lianshufeng/llm:unsloth lianshufeng/llm:LLaMA-Factory lianshufeng/llm:fastchat-cpu vllm/vllm-openai

REM docker-pull.exe -m docker.jpy.wang -cleanImage=true  lianshufeng/llm:unsloth lianshufeng/llm:vllm-cuda12.1 lianshufeng/llm:LLaMA-Factory lianshufeng/llm:fastchat-cpu

pause