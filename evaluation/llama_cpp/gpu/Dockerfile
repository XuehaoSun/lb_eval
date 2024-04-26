ARG CUDA_IMAGE="11.6.1-devel-ubuntu20.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    gcc wget software-properties-common && \
    add-apt-repository ppa:cnugteren/clblast && \
    apt update && \
    apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev && \
    mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tip && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

COPY . .

# Install depencencies
ENV PATH /opt/conda/bin:$PATH

# SHELL ["/bin/bash", "-c"]
SHELL ["/bin/bash", "--login", "-c"]

# RUN conda init bash
COPY requirements.txt /tmp/

RUN conda activate base && pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --requirement /tmp/requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip list

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1


# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Run the server
# CMD python3 -m llama_cpp.server
WORKDIR /
