FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    git-lfs \
    # NOTE: `ffmpeg` and `libmagic-dev` are required for audio related tasks
    # NOTE: `ffmpeg` version is constrained < 8, as it's a `torchcodec` requirement
    ffmpeg=7:* \
    libmagic-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# NOTE: `torchcodec` and hence the audio-related models as e.g. Wav2Vec, require
# both `libnpp` and `libnvrtc` to be preset for it to work seamlessly on CUDA
RUN if ! ldconfig -p | grep -q libnpp; then \
    apt-get update && apt-get install -y --no-install-recommends libnpp-dev; \
    fi && \
    if ! ldconfig -p | grep -q libnvrtc; then \
    apt-get update && apt-get install -y --no-install-recommends cuda-nvrtc-dev; \
    fi && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

RUN mkdir -p /home/huggingface && \
    useradd -d /home/huggingface -s /bin/bash huggingface && \
    chown -R huggingface:huggingface /home/huggingface

RUN mkdir -p /home/huggingface/.triton/autotune && \
    chown -R huggingface:huggingface /home/huggingface/.triton

ENV HOME=/home/huggingface
USER huggingface
WORKDIR /home/huggingface

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/huggingface/.cargo/bin:/home/huggingface/.local/bin/:$PATH"

RUN uv python install 3.11

RUN uv venv /home/huggingface/venv --python 3.11
ENV VIRTUAL_ENV=/home/huggingface/venv \
    PATH="/home/huggingface/venv/bin:$PATH"

ARG CUDA="cu128"
ARG PYTORCH="2.8.0"
ARG FLASH_ATTN="2.8.3"
ARG TRANSFORMERS="5.0.0"
ARG HUGGINGFACE_HUB="1.3.5"
# NOTE: Diffusers doesn't yet natively support Transformers v5.0.0
# https://github.com/huggingface/diffusers/pull/12976
ARG DIFFUSERS="0.36.0"
ARG TRL="0.27.1"
ARG PEFT="0.18.1"
ARG BITSANDBYTES="0.49.1"
ARG DATASETS="4.5.0"
ARG ACCELERATE="1.12.0"
ARG EVALUATE="0.4.6"
ARG SENTENCE_TRANSFORMERS="5.2.2"
ARG DEEPSPEED="0.18.5"
ARG KERNELS="0.12.1"

RUN uv pip install "torch==${PYTORCH}" torchvision torchaudio --torch-backend=${CUDA}

RUN uv pip install packaging ninja && \
    uv pip install "flash-attn==${FLASH_ATTN}" --no-build-isolation

RUN uv pip install --no-cache-dir \
    "transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS}" \
    "huggingface_hub[cli,hf-xet]==${HUGGINGFACE_HUB}" \
    "kernels==${KERNELS}" \
    "diffusers==${DIFFUSERS}" \
    "datasets==${DATASETS}" \
    "accelerate==${ACCELERATE}" \
    "evaluate==${EVALUATE}" \
    "peft==${PEFT}" \
    "trl[liger,peft,vlm]==${TRL}" \
    "sentence-transformers==${SENTENCE_TRANSFORMERS}" \
    "deepspeed==${DEEPSPEED}" \
    "bitsandbytes==${BITSANDBYTES}"

RUN uv pip install tensorboard jupyter notebook

RUN uv pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    google-cloud-pubsub \
    google-cloud-logging
