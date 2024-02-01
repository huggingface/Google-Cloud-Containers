FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG PYTORCH='2.0.0' 
ARG CUDA='cu118'
ARG PYTHON='3.10'
ARG TRANSFORMERS='4.35.0'
ARG DIFFUSERS='0.23.0'
ARG PEFT='0.7.0'
ARG TRL='0.7.4'
ARG BITSANDBYTES='0.41.3'
ARG DATASETS='2.14.0'
ARG ACCELERATE='0.23.0'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.0'
ARG DEEPSPEED='0.12.2'
ARG TENSORBOARD='2.15.1'
ARG FLASH_ATTN='2.3.0'
ARG NOTEBOOK='7.0.6'
ARG MARKUPSAFE='2.1.1'
ARG MAX_JOBS=4

RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
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


# Install python 3.10
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python${PYTHON} python3.10-distutils

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON}

# Set the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON} 1


# # update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

#Install PyTorch
RUN python3 -m pip install --no-cache-dir -U torch==${PYTORCH} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA}

# Install Hugging Face Libraries
RUN python3 -m pip install --upgrade --no-cache-dir \
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
  tensorboard==${TENSORBOARD} \
  notebook==${NOTEBOOK} \
  MarkupSafe==${MARKUPSAFE}

# Install Flash attention
RUN python3 -m pip install --no-cache-dir \
  packaging \
  ninja
RUN MAX_JOBS=${MAX_JOBS} python3 -m pip install flash-attn==${FLASH_ATTN} --no-build-isolation

# Check if all packages can be imported
RUN python3 -c "import transformers, diffusers, datasets, accelerate, evaluate, peft, trl, sentence_transformers, deepspeed, bitsandbytes, flash_attn; print('All packages installed successfully.')"
