FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG CUDA="cu124"
ARG PYTORCH="2.3.1"
ARG FLASH_ATTN="2.6.3"
ARG TRANSFORMERS="4.46.1"
ARG HUGGINGFACE_HUB="0.26.2"
ARG DIFFUSERS="0.31.0"
ARG PEFT="0.13.2"
ARG TRL="0.11.4"
ARG BITSANDBYTES="0.44.1"
ARG DATASETS="3.1.0"
ARG ACCELERATE="1.0.1"
ARG EVALUATE="0.4.3"
ARG SENTENCE_TRANSFORMERS="3.2.1"
ARG DEEPSPEED="0.15.3"
ARG MAX_JOBS=4

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
    libaio-dev \
    protobuf-compiler \
    python3.11 \
    python3.11-dev \
    libsndfile1-dev \
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip from source
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Upgrade pip and install uv
RUN pip install --upgrade pip && pip install --upgrade uv

# Install latest release PyTorch (PyTorch must be installed before any DeepSpeed c++/cuda ops.)
RUN uv pip install --no-cache-dir --system torch==${PYTORCH} torchvision torchaudio

# Upgrade FlashAttnV2
RUN uv pip install --no-cache-dir --system packaging ninja
RUN MAX_JOBS=${MAX_JOBS} uv pip install --no-cache-dir --no-build-isolation --system flash-attn==${FLASH_ATTN}

# Install Hugging Face Libraries
RUN uv pip install --no-cache-dir --system \
    "transformers[sklearn,sentencepiece,audio,vision]==${TRANSFORMERS}" \
    "huggingface_hub[hf_transfer]==${HUGGINGFACE_HUB}" \
    "diffusers==${DIFFUSERS}" \
    "datasets==${DATASETS}" \
    "accelerate==${ACCELERATE}" \
    "evaluate==${EVALUATE}" \
    "peft==${PEFT}" \
    "trl==${TRL}" \
    "sentence-transformers==${SENTENCE_TRANSFORMERS}" \
    "deepspeed==${DEEPSPEED}" \
    "bitsandbytes==${BITSANDBYTES}" \
    tensorboard \
    jupyter notebook

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

# Install Google Cloud Dependencies
RUN uv pip install --upgrade --no-cache-dir --system \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    google-cloud-pubsub \
    google-cloud-logging \
    "protobuf<4.0.0"
