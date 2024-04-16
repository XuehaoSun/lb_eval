ARG UBUNTU_VER=22.04
FROM ubuntu:${UBUNTU_VER} as devel

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    python3-setuptools \
    autoconf \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    numactl \
    time \
    wget \
    bc \
    vim

RUN ln -sf $(which python3) /usr/bin/python

RUN python -m pip --no-cache-dir install --upgrade pip
RUN python -m pip install --no-cache-dir setuptools wheel

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
RUN pip list

WORKDIR /