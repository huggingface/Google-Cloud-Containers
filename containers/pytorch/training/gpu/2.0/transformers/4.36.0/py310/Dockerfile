FROM nvcr.io/nvidia/pytorch:23.05-py3
# The image with PyTorch = 2.0.0, CUDA = 12.1.0, Python =  3.10.6"
# You can read more about it here: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-05.html#rel-23-05

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG TRANSFORMERS='4.36.0'
ARG DIFFUSERS='0.23.0'
ARG PEFT='0.7.0'
ARG TRL='0.7.4'
ARG BITSANDBYTES='0.42.0'
ARG DATASETS='2.14.0'
ARG ACCELERATE='0.23.0'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.0'
ARG DEEPSPEED='0.12.2'
ARG FLASH_ATTN='2.3.0'
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

# Install Flash attention
RUN pip install --no-cache-dir \
  packaging \
  ninja
RUN MAX_JOBS=${MAX_JOBS} pip install flash-attn==${FLASH_ATTN} --no-build-isolation

# Remove TransformerEngine for using Flash-AttnV2 as the base image comes with TransformerEngine-0.8 whish is not compatible with Flash-AttnV2
RUN pip uninstall -y transformer-engine

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

# Check if correct versions are installed
RUN python -c "import transformers, diffusers, datasets, accelerate, evaluate, peft, trl, sentence_transformers, deepspeed, bitsandbytes, flash_attn; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (accelerate, '${ACCELERATE}'), (evaluate, '${EVALUATE}'), (peft, '${PEFT}'), (trl, '${TRL}'), \
                (sentence_transformers, '${SENTENCE_TRANSFORMERS}'), (deepspeed, '${DEEPSPEED}'), (bitsandbytes, '${BITSANDBYTES}'), \
                (flash_attn, '${FLASH_ATTN}')]])"