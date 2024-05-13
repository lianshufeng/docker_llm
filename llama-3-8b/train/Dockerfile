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

