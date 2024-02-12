FROM nvcr.io/nvidia/pytorch:23.10-py3
# The image with PyTorch = 2.1.0a0+32f93b1, CUDA = 12.2.2, Python =  3.10.12"

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions

#JAX Compatible with CUDA 12.2 
ARG JAX='0.4.23' 
ARG FLAX='0.8.1'
ARG TRANSFORMERS='4.37.2'
ARG DIFFUSERS='0.26.1'
ARG PEFT='0.8.2'
ARG TRL='0.7.10'
ARG BITSANDBYTES='0.42.0'
ARG DATASETS='2.16.1'
ARG ACCELERATE='0.27.0'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.1'
ARG DEEPSPEED='0.13.1'
ARG FLASH_ATTN='2.5.2'
ARG MAX_JOBS=4

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


# update pip
RUN pip install --no-cache-dir --upgrade pip

# Uninstall Transformer Engine
RUN pip uninstall -y transformer-engine

# Upgrade FlashAttnV2
RUN pip install --no-cache-dir \
  packaging \
  ninja
RUN MAX_JOBS=${MAX_JOBS} pip install flash-attn==${FLASH_ATTN} --no-build-isolation

# Install JAX
RUN pip install --upgrade jax==${JAX} jaxlib==${JAX}+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Flax
RUN pip install --upgrade --no-cache-dir flax==${FLAX}

#Install Hugging Face Libraries
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
  bitsandbytes==${BITSANDBYTES}

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


# Check if correct versions are installed
RUN python -c "import transformers, diffusers, datasets, accelerate, evaluate, peft, trl, sentence_transformers, deepspeed, bitsandbytes, flash_attn, jax, flax; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (accelerate, '${ACCELERATE}'), (evaluate, '${EVALUATE}'), (peft, '${PEFT}'), (trl, '${TRL}'), \
                (sentence_transformers, '${SENTENCE_TRANSFORMERS}'), (deepspeed, '${DEEPSPEED}'), (bitsandbytes, '${BITSANDBYTES}'), \
                (jax, '${JAX}'), (flax, '${FLAX}')]])" 