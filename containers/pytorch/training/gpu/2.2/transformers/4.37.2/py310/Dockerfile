FROM nvcr.io/nvidia/pytorch:24.01-py3
# The image with PyTorch = 2.2.0a0+81ea7a4, CUDA = 12.3.2, Python =  3.10.12"
# You can read more about it here: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01
# FlashAttnV2 = 2.0.4 supported out of the box with this image

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG TRANSFORMERS='4.37.2'
ARG DIFFUSERS='0.26.1'
ARG PEFT='0.8.2'
ARG TRL='0.7.10'
ARG BITSANDBYTES='0.42.0'
ARG DATASETS='2.16.1'
ARG ACCELERATE='0.26.1'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.1'
ARG DEEPSPEED='0.13.1'
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
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Update pip
RUN pip install --upgrade pip

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
  bitsandbytes==${BITSANDBYTES}

# # Check if correct versions are installed
RUN python -c "import transformers, diffusers, datasets, accelerate, evaluate, peft, trl, sentence_transformers, deepspeed, bitsandbytes, flash_attn, torch; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (accelerate, '${ACCELERATE}'), (evaluate, '${EVALUATE}'), (peft, '${PEFT}'), (trl, '${TRL}'), \
                (sentence_transformers, '${SENTENCE_TRANSFORMERS}'), (deepspeed, '${DEEPSPEED}'), (bitsandbytes, '${BITSANDBYTES}'), \
                (torch, '2.2.0a0+81ea7a4'), (flash_attn, '2.0.4')]])"