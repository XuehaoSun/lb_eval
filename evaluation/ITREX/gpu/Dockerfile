FROM ubuntu:22.04 as devel

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    autoconf \
    build-essential \
    git \
    numactl \
    time \
    wget \
    vim

RUN ln -sf $(which python3) /usr/bin/python

RUN pip --no-cache-dir install --upgrade pip
RUN pip install --no-cache-dir setuptools==69.5.1 wheel

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip list

WORKDIR /