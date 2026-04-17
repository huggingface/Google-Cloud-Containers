FROM ubuntu:24.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    git \
    gnupg \
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

# NOTE: anthoscli is not required for the intended use of gcloud SDK within this
# context, hence we're safe to remove it, preventing the following CVEs:
# - CVE-2025-68121
# - CVE-2026-27143
# - CVE-2026-33186
# Which are originated due to the bundled Go version in the pre-compiled anthoscli
RUN rm -rf /usr/lib/google-cloud-sdk/bin/anthoscli

# NOTE: nic_sampler is not required for inference workloads, and is removed to
# prevent CVEs originating from the bundled Go version used to compile it
# - CVE-2025-22871
# - CVE-2026-33186
# - CVE-2024-24790
# - CVE-2026-27143
# - CVE-2025-68121
RUN rm -f /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/plugins/efa_metrics/nic_sampler

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

WORKDIR /home/huggingface

# NOTE: Until `hf-serve` is made public, the extra `cpu` needs to be installed
# manually before installing `hf-serve` from the wheel
RUN uv pip install "torch==2.7.1" "torchvision" "torchcodec<0.6" --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=huggingface:huggingface containers/pytorch/inference/cpu/2.7.1/transformers/4.57.3/py311/hf_serve-0.0.1rc7-py3-none-any.whl .
RUN uv pip install "hf_serve-0.0.1rc7-py3-none-any.whl"

RUN uv pip install google-cloud-storage crcmod --upgrade

COPY --chown=huggingface:huggingface containers/pytorch/inference/cpu/2.7.1/transformers/4.57.3/py311/entrypoint.sh .
RUN chmod +x /home/huggingface/entrypoint.sh

USER huggingface
ENTRYPOINT ["/home/huggingface/entrypoint.sh"]
