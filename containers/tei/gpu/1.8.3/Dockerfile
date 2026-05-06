FROM alpine AS text-embeddings-inference

RUN mkdir -p /text-embeddings-inference
ADD https://github.com/huggingface/text-embeddings-inference/archive/refs/tags/v1.8.3.tar.gz /text-embeddings-inference/sources.tar.gz
RUN tar -C /text-embeddings-inference -xf /text-embeddings-inference/sources.tar.gz --strip-components=1

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"
# NOTE: `CARGO_CHEF` version is aligned with `cargo-chef` version installed in
# `lukemathwalker/cargo-chef:latest-rust-1.75-bookworm`
ENV CARGO_CHEF=0.1.71

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --version $CARGO_CHEF --locked

FROM base-builder AS planner

WORKDIR /usr/src

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN cargo chef prepare --recipe-path recipe.json

FROM base-builder AS builder

WORKDIR /usr/src

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --features google --recipe-path recipe.json && sccache -s

FROM builder AS builder-75

RUN CUDA_COMPUTE_CAP=75 cargo chef cook --release --features candle-cuda-turing,google --recipe-path recipe.json && sccache -s

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN CUDA_COMPUTE_CAP=75 cargo build --release --bin text-embeddings-router --features candle-cuda-turing,google && sccache -s

FROM builder AS builder-80

RUN CUDA_COMPUTE_CAP=80 cargo chef cook --release --features candle-cuda,google --recipe-path recipe.json && sccache -s

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN CUDA_COMPUTE_CAP=80 cargo build --release --bin text-embeddings-router --features candle-cuda,google && sccache -s

FROM builder AS builder-90

RUN CUDA_COMPUTE_CAP=90 cargo chef cook --release --features candle-cuda,google --recipe-path recipe.json && sccache -s

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN CUDA_COMPUTE_CAP=90 cargo build --release --bin text-embeddings-router --features candle-cuda,google && sccache -s

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

ENV HUGGINGFACE_HUB_CACHE=/tmp \
    PORT=8080 \
    USE_FLASH_ATTENTION=True \
    PATH="/usr/local/nvidia/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-75 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-75
COPY --from=builder-80 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-80
COPY --from=builder-90 /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router-90

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

COPY --chmod=775 containers/tei/gpu/1.8.3/entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]
