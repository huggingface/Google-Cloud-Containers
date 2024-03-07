FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
# The image with CUDA = 12.2.2

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG CUDA="12"
ARG CUDNN="89"
ARG TRANSFORMERS='4.38.1'
ARG DIFFUSERS='0.26.1'
ARG DATASETS='2.16.1'
# jax and flax compatible with transformers ["jax>=0.4.1,<=0.4.13", "jaxlib>=0.4.1,<=0.4.13", "flax>=0.4.1,<=0.7.0"] as mentioned in setup.py
ARG JAXLIB='0.4.13'

RUN apt-get update \
    && apt-get install -y \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    libaio-dev \
    # audio
    libsndfile1-dev \
    ffmpeg \
    apt-transport-https \
    gnupg \
    ca-certificates \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 python3.10-distutils python-is-python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && apt-get clean autoremove --yes 

# Set Python 3.10 as the default python version and upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    pip install --no-cache-dir --upgrade pip

#Install Hugging Face Libraries and JAXLIB with CUDA support
RUN pip install --upgrade --no-cache-dir \
  transformers[flax,sentencepiece]==${TRANSFORMERS} \
  diffusers==${DIFFUSERS} \
  datasets==${DATASETS} \
  jaxlib==${JAXLIB}+cuda${CUDA}.cudnn${CUDNN} -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Google Cloud Dependencies
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

RUN pip install --upgrade --no-cache-dir \
  google-cloud-storage \ 
  google-cloud-bigquery \
  google-cloud-aiplatform \
  google-cloud-pubsub \
  google-cloud-logging

#Check if correct versions are installed
RUN python -c "import transformers, diffusers, datasets, sentencepiece, jax, jaxlib, flax; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (jax, '0.4.13'), (jaxlib, '${JAXLIB}'), (flax, '0.7.0')]])"