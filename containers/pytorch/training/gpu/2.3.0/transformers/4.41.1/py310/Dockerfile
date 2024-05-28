FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG CUDA='cu121'
ARG PYTORCH='2.3.0'
ARG FLASH_ATTN='2.5.8'
ARG TRANSFORMERS='4.41.1'
ARG DIFFUSERS='0.28.0'
ARG PEFT='0.11.1'
ARG TRL='0.8.6'
ARG BITSANDBYTES='0.43.1'
ARG DATASETS='2.19.1'
ARG ACCELERATE='0.30.1'
ARG EVALUATE='0.4.2'
ARG SENTENCE_TRANSFORMERS='2.7.0'
ARG DEEPSPEED='0.14.2'
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
        python3-dev \
        python3-pip \
        python3.11 \
        libsndfile1-dev \
        ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Update pip
RUN pip install --upgrade pip

# Install latest release PyTorch (PyTorch must be installed before any DeepSpeed c++/cuda ops.)
RUN pip install --no-cache-dir -U torch==${PYTORCH} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}

# Upgrade FlashAttnV2
RUN pip install --no-cache-dir \
  packaging \
  ninja
RUN MAX_JOBS=${MAX_JOBS} pip install flash-attn==${FLASH_ATTN} --no-build-isolation

# Install Hugging Face Libraries
RUN pip install --upgrade --no-cache-dir \
  transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
  diffusers==${DIFFUSERS} \
  datasets==${DATASETS} \
  accelerate==${ACCELERATE} \
  evaluate==${EVALUATE} \
  peft==${PEFT} \
  trl==${TRL} \
  sentence-transformers==${SENTENCE_TRANSFORMERS} \
  deepspeed==${DEEPSPEED} \
  bitsandbytes==${BITSANDBYTES} \
  tensorboard \
  jupyter notebook

# Install Google Cloud Dependencies
RUN pip install --upgrade --no-cache-dir \
  google-cloud-storage \ 
  google-cloud-bigquery \
  google-cloud-aiplatform \
  google-cloud-pubsub \
  google-cloud-logging \
  "protobuf<4.0.0"

# ENTRYPOINT ["/bin/bash"]