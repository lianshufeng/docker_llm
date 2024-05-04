# ARG from=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# FROM ${from} as base


FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# python3
# RUN apt-get update && apt-get install -y python3.10 python3-pip
# torch
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache
# llama factory requirements
# RUN pip3 install transformers==4.37.2 datasets==2.16.1 accelerate==0.25.0 peft==0.7.1 trl==0.7.10 gradio==3.50.2 \
    # deepspeed modelscope ipython scipy einops sentencepiece protobuf jieba rouge-chinese nltk sse-starlette matplotlib \
    # --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
# unsloth
# RUN apt-get install -y git
# RUN pip install --upgrade pip
# RUN pip install triton --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install "unsloth[cu121_ampere_torch220] @ git+https://github.com/unslothai/unsloth.git"


RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    vim \
	gcc \
	g++ \
&& rm -rf /var/lib/apt/lists/*


# pip
RUN pip install --upgrade pip

# torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache

# unsloth
RUN pip3 install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
# For Pytorch 2.2.0: Use the "ampere" path for newer RTX 30xx GPUs or higher.
# RUN pip3 install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"

# major_version >8
# Use this for new GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)
# RUN pip3 install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

# Use this for older GPUs (V100, Tesla T4, RTX 20xx)
RUN pip3 install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

