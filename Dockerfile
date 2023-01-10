FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN set -xe && \
    apt update && \
    apt install -y \
        git wget \
        build-essential \
        ffmpeg \
        libopencv-dev \
        python-is-python3

RUN apt install -y python3-distutils python3.9 python3.9-dev \
    && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.9 ~/get-pip.py \
    && rm ~/get-pip.py \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools --no-cache-dir \
    && python -m pip install wheel --no-cache-dir

RUN git clone -b st-gcn https://github.com/Jeong-Labo/st-gcn.git && \
    pip3 install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip3 install -r /app/st-gcn/requirements.txt && \
    cp -r /app/st-gcn/torchlight/torchlight /usr/local/lib/python3.8/dist-packages/ && \
    cd /app/st-gcn && \
    bash tools/get_models.sh
