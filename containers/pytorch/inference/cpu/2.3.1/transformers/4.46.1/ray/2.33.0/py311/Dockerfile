FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install required dependencies
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
    build-essential \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    python3.11 \
    python3.11-dev \
    libsndfile1-dev \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip from source
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Hugging Face Inference Toolkit
ARG HF_INFERENCE_TOOLKIT_VERSION=0.5.2
ARG HF_INFERENCE_TOOLKIT_URL=git+https://github.com/huggingface/huggingface-inference-toolkit.git@${HF_INFERENCE_TOOLKIT_VERSION}
RUN pip install --upgrade "huggingface-inference-toolkit[torch,diffusers,st,google] @ ${HF_INFERENCE_TOOLKIT_URL}" --no-cache-dir

# Install Ray and python-json-logger
# https://cloud.google.com/vertex-ai/docs/open-source/ray-on-vertex-ai/create-cluster#custom-image
RUN pip install --upgrade "ray[all]==2.33.0" "python-json-logger==2.0.7" --no-cache-dir

ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Install Google CLI single command
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Copy entrypoint and change permissions
COPY --chmod=0755 containers/pytorch/inference/cpu/2.3.1/transformers/4.46.1/ray/2.33.0/py311/entrypoint.sh entrypoint.sh
ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
