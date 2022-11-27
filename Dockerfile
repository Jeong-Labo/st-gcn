FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
    && apt install -y wget curl git make build-essential

RUN apt install -y python3-distutils python3.9 python3.9-dev \
    && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.9 ~/get-pip.py \
    && rm ~/get-pip.py \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools --no-cache-dir \
    && python -m pip install wheel --no-cache-dir

RUN pip install jupyter notebook jupyterlab jupyter-contrib-nbextensions

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN apt install -y sudo libopencv-dev qtbase5-dev cmake-qt-gui \
    && git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose \
    && cd openpose \
    && bash ./scripts/ubuntu/install_deps.sh \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j`nproc` \
    && make install \
    && cd /app

COPY torchlight/ ./torchlight/

RUN cd torchlight \
    && python setup.py install --install-purelib /usr/local/lib/python3.9/dist-packages \
    && cd ../
