FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

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
    python3-dev \
    python3-pip \
    python3.11 \
    libsndfile1-dev \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face Inference Toolkit
ARG HF_INFERENCE_TOOLKIT_VERSION=0.4.3
ARG HF_INFERENCE_TOOLKIT_URL=git+https://github.com/huggingface/huggingface-inference-toolkit.git@${HF_INFERENCE_TOOLKIT_VERSION}
RUN pip install "${HF_INFERENCE_TOOLKIT_URL}#egg=huggingface-inference-toolkit[torch,diffusers,st,google]"

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
COPY --chmod=0755 containers/pytorch/inference/cpu/2.2.2/transformers/4.44.0/py311/entrypoint.sh entrypoint.sh
ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
