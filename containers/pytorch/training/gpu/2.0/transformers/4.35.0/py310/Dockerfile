FROM nvcr.io/nvidia/pytorch:23.05-py3
LABEL maintainer="Hugging Face"

# Versions
ARG PYTORCH='2.0.0' # Just for reference 
ARG CUDA='12.1.1' # Copied from https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-05.html#rel-23-05
ARG PYTHON='3.10.6' # For reference, not used for installing
ARG TRANSFORMERS='4.35.0'
ARG DIFFUSERS='0.23.0'
ARG PEFT='0.6.0'
ARG TRL='0.7.4'
ARG BITSANDBYTES='0.41.3'
ARG OPTIMUM='1.14.0'
ARG DATASETS='2.14.0'
ARG ACCELERATE='0.23.0'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.0'
ARG DEEPSPEED='0.12.2'
ARG TENSORBOARD='2.15.1'
ARG REQUESTS='2.31.0'
ARG NOTEBOOK='7.0.6'
ARG MARKUPSAFE='2.1.1'
ARG JINJA2='3.1.2'
ARG ATTRS='23.1.0'
ARG FLASH_ATTN='2.3.0'

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Upgrade Tensorboard
RUN python3 -m pip install --no-cache-dir tensorboard==${TENSORBOARD}

# Install DeepSpeed
RUN python3 -m pip install --no-cache-dir deepspeed==${DEEPSPEED}

# Install bitsandbytes
RUN python3 -m pip install --no-cache-dir bitsandbytes==${BITSANDBYTES}

# Install Hugging Face Libraries
RUN python3 -m pip install --upgrade --no-cache-dir \
  transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
  diffusers==${DIFFUSERS} \
  datasets==${DATASETS} \
  accelerate==${ACCELERATE} \
  evaluate==${EVALUATE} \
  peft==${PEFT} \
  trl==${TRL} \
  optimum==${OPTIMUM} \
  requests==${REQUESTS} \
  notebook==${NOTEBOOK} \
  MarkupSafe==${MARKUPSAFE} \
  Jinja2==${JINJA2} \
  attrs==${ATTRS}

# Install Sentence Transformers
RUN python3 -m pip install --no-cache-dir sentence-transformers==${SENTENCE_TRANSFORMERS}

# Install Flash attention
RUN python3 -m pip install --no-cache-dir \
  packaging \
  ninja
RUN MAX_JOBS=4 python3 -m pip install flash-attn==${FLASH_ATTN} --no-build-isolation