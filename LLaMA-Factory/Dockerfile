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


RUN git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory /app


WORKDIR /app


RUN pip install -r requirements.txt

COPY . /app/
RUN pip install -e .[metrics,bitsandbytes,qwen]

VOLUME [ "/root/.cache/huggingface/", "/app/data", "/app/output" ]
EXPOSE 7860


# fix 
RUN pip install auto_gptq>=0.5.0 optimum autoawq

CMD [ "llamafactory-cli", "webui" ]

