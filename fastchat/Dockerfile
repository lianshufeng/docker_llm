FROM ubuntu:22.04 as base


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

# ln python3
RUN ln -s $(which pip3) /usr/local/bin/pip && ln -s $(which python3) /usr/local/bin/python

# torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache

# transformers 
RUN pip install git+https://github.com/huggingface/transformers


# fschat pydantic 
RUN pip install "fschat[model_worker]"==0.2.36 --no-cache

#fix pydantic
RUN pip uninstall pydantic -y 
RUN pip install pydantic==1.10.11 --no-cache

# 
#RUN pip install optimum auto-gptq


