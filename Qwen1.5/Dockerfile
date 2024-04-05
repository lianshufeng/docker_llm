ARG CUDA_VERSION=12.1.0
ARG RUNTIME=runtime # runtime or devel
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn8-${RUNTIME}-ubuntu22.04

FROM ${from} as base


RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    vim \
&& rm -rf /var/lib/apt/lists/*

# torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache

# transformers 
# RUN pip3 install transformers>=4.37.0 --no-cache
RUN pip install git+https://github.com/huggingface/transformers

# vLLM
RUN pip3 install vLLM>=0.3.0 --no-cache

# fschat
RUN pip3 install --no-cache fschat[model_worker]==0.2.36

# gptq
RUN pip3 install --no-cache "optimum==1.17.1" "auto-gptq==0.7.1" 