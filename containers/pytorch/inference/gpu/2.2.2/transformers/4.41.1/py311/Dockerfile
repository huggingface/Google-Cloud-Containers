FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
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

# install python packages
RUN pip install --no-cache-dir -U pip "torch==2.2.2" "torchvision" "torchaudio" "sentence_transformers==2.7.0" "diffusers==0.26.3" "accelerate==0.27.2" google-cloud-storage

# Hugging Face Inference Toolkit
COPY containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/huggingface_inference_toolkit-0.4.1-py3-none-any.whl /app/huggingface_inference_toolkit-0.4.1-py3-none-any.whl 
RUN pip install huggingface_inference_toolkit-0.4.1-py3-none-any.whl
RUN rm /app/huggingface_inference_toolkit-0.4.1-py3-none-any.whl

# copy entrypoint and change permissions
COPY --chmod=0755 containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/entrypoint.sh entrypoint.sh

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]