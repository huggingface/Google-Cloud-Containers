FROM ubuntu:24.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    # NOTE: `ffmpeg` and `libmagic-dev` are required for audio related tasks
    # NOTE: `ffmpeg` version is constrained < 8, as it's a `torchcodec` requirement
    ffmpeg=7:* \
    libmagic-dev \
    # NOTE: `espeak-ng` is the backend used by `phonemizer` so adding it here only
    # tentatively as it's most likely "too specific" for models relying on `phonemizer`
    # as e.g. https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
    espeak-ng \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# NOTE: Inference Endpoints API writes the Hugging Face Hub repository in
# `/repository` hence it should allow any user to read from it
RUN mkdir -p /repository && chmod 755 /repository

# NOTE: GID and UID set to 1001 instead of standard 1000, given that's reserved
# for the default non-root Ubuntu user
RUN groupadd --gid 1001 huggingface \
    && useradd --uid 1001 --gid huggingface --shell /bin/bash --create-home huggingface

# Create the /opt/huggingface directory and set correct owner and permissions,
# given that's the directory formerly used for the Vertex AI DLCs
RUN mkdir -p /opt/huggingface \
    && chown 1001:1001 /opt/huggingface \
    && chmod 755 /opt/huggingface

USER huggingface
WORKDIR /home/huggingface

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/huggingface/.cargo/bin:/home/huggingface/.local/bin/:$PATH"

RUN uv python install 3.12

RUN uv venv /home/huggingface/venv --python 3.12
ENV VIRTUAL_ENV=/home/huggingface/venv \
    PATH="/home/huggingface/venv/bin:$PATH"

WORKDIR /home/huggingface/app

COPY hf_serve-0.0.1rc6-py3-none-any.whl .
RUN uv pip install --frozen hf_serve-0.0.1rc6-py3-none-any.whl --extra cpu

RUN uv pip install google-cloud-storage crcmod --upgrade

COPY --chown=huggingface:huggingface entrypoint.sh /home/huggingface/entrypoint.sh
RUN chmod +x /home/huggingface/entrypoint.sh

USER huggingface
ENTRYPOINT ["/home/huggingface/entrypoint.sh"]
