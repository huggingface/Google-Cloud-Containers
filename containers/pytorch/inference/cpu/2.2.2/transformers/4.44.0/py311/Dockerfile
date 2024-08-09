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
    ffmpeg \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Hugging Face Inference Toolkit
ARG HF_INFERENCE_TOOLKIT_WHL="huggingface_inference_toolkit-0.4.2-py3-none-any.whl"
COPY containers/pytorch/inference/cpu/2.2.2/transformers/4.44.0/py311/${HF_INFERENCE_TOOLKIT_WHL} /app/${HF_INFERENCE_TOOLKIT_WHL}

RUN pip install "${HF_INFERENCE_TOOLKIT_WHL}[torch,diffusers,st,google]" && \
    rm /app/${HF_INFERENCE_TOOLKIT_WHL}

ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# copy entrypoint and change permissions
COPY --chmod=0755 containers/pytorch/inference/cpu/2.2.2/transformers/4.44.0/py311/entrypoint.sh entrypoint.sh
ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
