# Fetch and extract the TEI sources
FROM alpine AS tei
RUN mkdir -p /tei
ADD https://github.com/huggingface/text-embeddings-inference/archive/refs/tags/v1.2.2.tar.gz /tei/sources.tar.gz
RUN tar -C /tei -xf /tei/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TEI original Dockerfile)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"
# aligned with `cargo-chef` version in `lukemathwalker/cargo-chef:latest-rust-1.75-bookworm` 
ENV CARGO_CHEF=0.1.62

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --version $CARGO_CHEF --locked

FROM base-builder AS planner

WORKDIR /usr/src

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM base-builder AS builder

ARG GIT_SHA
ARG DOCKER_LABEL
ARG VERTEX

# sccache specific variables
ARG ACTIONS_CACHE_URL
ARG ACTIONS_RUNTIME_TOKEN
ARG SCCACHE_GHA_ENABLED

WORKDIR /usr/src

COPY --from=planner /usr/src/recipe.json recipe.json

FROM builder as builder-75

RUN CUDA_COMPUTE_CAP=75 cargo chef cook --release --features google --features candle-cuda-turing --features http --no-default-features --recipe-path recipe.json && sccache -s

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

RUN CUDA_COMPUTE_CAP=75 cargo build --release --bin text-embeddings-router -F candle-cuda-turing -F http -F google --no-default-features && sccache -s


FROM builder as builder-80

RUN CUDA_COMPUTE_CAP=80 cargo chef cook --release --features google --features candle-cuda --features http --no-default-features --recipe-path recipe.json && sccache -s

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

RUN CUDA_COMPUTE_CAP=80 cargo build --release --bin text-embeddings-router -F candle-cuda -F http -F google --no-default-features && sccache -s

FROM builder as builder-90

RUN CUDA_COMPUTE_CAP=90 cargo chef cook --release --features google --features candle-cuda --features http --no-default-features --recipe-path recipe.json && sccache -s

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

RUN CUDA_COMPUTE_CAP=90 cargo build --release --bin text-embeddings-router -F candle-cuda -F http -F google --no-default-features && sccache -s

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base

ARG DEFAULT_USE_FLASH_ATTENTION=True

ENV HUGGINGFACE_HUB_CACHE=/tmp \
    PORT=8080 \
    USE_FLASH_ATTENTION=$DEFAULT_USE_FLASH_ATTENTION

COPY --from=builder-75 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-75
COPY --from=builder-80 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-80
COPY --from=builder-90 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-90

COPY --from=tei /tei/cuda-all-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]
