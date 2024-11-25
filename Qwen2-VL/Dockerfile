FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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

# pip
RUN pip install --upgrade pip

# torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache


WORKDIR /opt/app

ADD ./requirements.txt /requirements.txt
ADD ./src /opt/app


RUN pip install -r /requirements.txt

RUN pip install -U flash-attn --no-build-isolation
