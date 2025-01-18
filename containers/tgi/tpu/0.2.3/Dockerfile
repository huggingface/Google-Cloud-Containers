# Enable GCP integration by default
ARG ENABLE_GOOGLE_FEATURE=1

# Fetch and extract the TGI sources
FROM alpine AS tgi
# TGI version 2.4.1 by default
ARG TGI_VERSION=v2.4.1
RUN test -n ${TGI_VERSION:?}
RUN mkdir -p /tgi
ADD https://github.com/huggingface/text-generation-inference/archive/${TGI_VERSION}.tar.gz /tgi/sources.tar.gz
RUN tar -C /tgi -xf /tgi/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TGI original Dockerfile)
# Note that the build image is aligned on the same Linux version as the base image (Debian bookworm/ Ubuntu 22.04)
FROM lukemathwalker/cargo-chef:latest-rust-1.80.1-bookworm AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef AS planner
COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ARG ENABLE_GOOGLE_FEATURE
RUN echo "Google Feature Status: ${ENABLE_GOOGLE_FEATURE}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.11-dev
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --profile release-opt --recipe-path recipe.json

COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
RUN if [ -n "$ENABLE_GOOGLE_FEATURE" ]; then \
        cargo build --profile release-opt --features google; \
    else \
        cargo build --profile release-opt; \
    fi

# Python base image
FROM ubuntu:22.04 AS base

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip3 --no-cache-dir install --upgrade pip

ARG ENABLE_GOOGLE_FEATURE
ARG VERSION='0.2.3'
RUN test -n ${VERSION:?}

FROM base AS optimum-tpu-installer

COPY . /tmp/src

RUN if [ -n "$ENABLE_GOOGLE_FEATURE" ]; then \
        # If we are building for GCP, we need to clone the optimum-tpu repo as this is built from the huggingface/Google-Cloud-Containers repository and not the huggingface/optimum-tpu repository
        git clone https://github.com/huggingface/optimum-tpu.git /opt/optimum-tpu && \
        cd /opt/optimum-tpu && git checkout v${VERSION}; \
    fi && \
        # Check if the optimum-tpu repo is cloned properly
        cp -a /tmp/src /opt/optimum-tpu && \
        if [ ! -d "/opt/optimum-tpu/optimum" ]; then \
            echo "Error: Building from incorrect repository. This build must be run from optimum-tpu repo. If building from google-cloud-containers repo, set ENABLE_GOOGLE_FEATURE=1 to automatically clone optimum-tpu" && \ 
            exit 1; \
    fi


# Python server build image
FROM base AS pyserver

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    make \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN install -d /pyserver
WORKDIR /pyserver
COPY --from=optimum-tpu-installer /opt/optimum-tpu/text-generation-inference/server server
COPY --from=tgi /tgi/proto proto
RUN pip3 install -r server/build-requirements.txt
RUN VERBOSE=1 BUILDDIR=/pyserver/build PROTODIR=/pyserver/proto VERSION=${VERSION} make -C server gen-server

# TPU base image (used for deployment)
FROM base AS tpu_base

ARG VERSION=${VERSION}

# Install system prerequisites
# NOTE: we need both python3.10 and python3.11 to be installed, as the TGI router uses python 3.11 and optimum-tpu uses
# python 3.10. This has been fixed on newest version of optimum-tpu and will be removed in the next version (see
# https://github.com/huggingface/optimum-tpu/pull/135 for details).
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    libpython3.10 \
    libpython3.11 \
    python3.11 \
    git \
    gnupg2 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update pip
RUN pip install --upgrade pip

# Install HuggingFace packages
ARG TRANSFORMERS_VERSION='4.46.3'
ARG ACCELERATE_VERSION='1.1.1'
ARG SAFETENSORS_VERSION='0.4.5'

ARG ENABLE_GOOGLE_FEATURE

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VERSION=${VERSION}

ENV PORT=${ENABLE_GOOGLE_FEATURE:+8080}
ENV PORT=${PORT:-80}

ENV HF_HOME=${ENABLE_GOOGLE_FEATURE:+/tmp}
ENV HF_HOME=${HF_HOME:-/data}

# Install requirements for TGI, that uses python3.11
RUN python3.11 -m pip install transformers==${TRANSFORMERS_VERSION}

# Install requirements for optimum-tpu, then for TGI then optimum-tpu
RUN python3 -m pip install hf_transfer safetensors==${SAFETENSORS_VERSION} typer
COPY --from=optimum-tpu-installer /opt/optimum-tpu /opt/optimum-tpu
RUN python3 /opt/optimum-tpu/optimum/tpu/cli.py install-jetstream-pytorch --yes
RUN python3 -m pip install -e /opt/optimum-tpu \
        -f https://storage.googleapis.com/libtpu-releases/index.html

# Install benchmarker
COPY --from=builder /usr/src/target/release-opt/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release-opt/text-generation-router-v2 /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release-opt/text-generation-launcher /usr/local/bin/text-generation-launcher
# Install python server
COPY --from=pyserver /pyserver/build/dist dist
RUN pip install dist/text_generation_server*.tar.gz


# TPU compatible image for Inference Endpoints
FROM tpu_base AS inference-endpoint

COPY text-generation-inference/docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

FROM tpu_base AS google-cloud-containers

# Install Google specific components if ENABLE_GOOGLE_FEATURE is set
RUN if [ -n "$ENABLE_GOOGLE_FEATURE" ]; then \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            git && \
        rm -rf /var/lib/apt/lists/* && \
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
            | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
        apt-get update -y && \
        apt-get install google-cloud-sdk -y; \
    fi

# Custom entrypoint for Google
COPY --chmod=775 containers/tgi/tpu/${VERSION}/entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]