FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    git-lfs \
    ffmpeg=7:* \
    libmagic-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

RUN uv pip install "torch==2.11.0" torchvision torchaudio --torch-backend=cu126

RUN uv pip install --no-cache-dir \
    "transformers[audio,chat_template,kernels,video,vision]>=5.13.1"
    "huggingface_hub[hf_xet]==1.23.0" \
    "datasets==5.0.0" \
    "teich==0.1.5" \
    "accelerate==1.14.0" \
    "trl[deepspeed,peft,quantization,vlm]==1.8.0"

RUN uv pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    google-cloud-pubsub \
    google-cloud-logging \
    gcsfs
