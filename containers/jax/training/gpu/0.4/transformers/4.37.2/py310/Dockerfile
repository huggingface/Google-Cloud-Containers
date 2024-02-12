FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
# The image with CUDA = 12.2.2

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions

#JAX Compatible with CUDA 12.2 
ARG JAX='0.4.23' 
ARG FLAX='0.8.1'
ARG TRANSFORMERS='4.37.2'
ARG DIFFUSERS='0.26.1'
ARG DATASETS='2.16.1'


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
    && apt-get clean autoremove --yes 


# Install python 3.10
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.10 python3.10-distutils python-is-python3

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# update pip
RUN pip install --no-cache-dir --upgrade pip

# Install JAX
RUN pip install --upgrade jax==${JAX} jaxlib==${JAX}+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Flax
RUN pip install --upgrade --no-cache-dir flax==${FLAX}

#Install Hugging Face Libraries
RUN pip install --upgrade --no-cache-dir \
  transformers==${TRANSFORMERS} \
  diffusers==${DIFFUSERS} \
  datasets==${DATASETS} 

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
RUN python -c "import transformers, diffusers, datasets, jax, flax; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (jax, '${JAX}'), (flax, '${FLAX}')]])" 