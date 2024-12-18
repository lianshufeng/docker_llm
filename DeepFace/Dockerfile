FROM tensorflow/tensorflow:2.11.0-gpu

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip git zip unzip wget curl htop gcc libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
ADD ./ ./


RUN python -m pip install --upgrade pip
#RUN pip install --ignore-installed blinker
RUN pip install -r requirements.txt

CMD ["python", "/opt/app/src/api.py"]