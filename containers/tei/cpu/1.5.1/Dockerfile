# Fetch and extract the TGI sources
FROM alpine AS tei

RUN mkdir -p /tei
ADD https://github.com/huggingface/text-embeddings-inference/archive/refs/tags/v1.5.1.tar.gz /tei/sources.tar.gz
RUN tar -C /tei -xf /tei/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TEI original Dockerfile)
FROM lukemathwalker/cargo-chef:latest-rust-1.75-bookworm AS chef
WORKDIR /usr/src

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

# Donwload, configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --features ort --no-default-features --recipe-path recipe.json && sccache -s

COPY --from=tei /tei/backends backends
COPY --from=tei /tei/core core
COPY --from=tei /tei/router router
COPY --from=tei /tei/Cargo.toml ./
COPY --from=tei /tei/Cargo.lock ./

FROM builder AS http-builder

RUN cargo build --release --bin text-embeddings-router -F google -F ort -F http --no-default-features && sccache -s

FROM builder AS grpc-builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=tei /tei/proto proto

RUN cargo build --release --bin text-embeddings-router -F google -F grpc -F ort --no-default-features && sccache -s

FROM debian:bookworm-slim AS base

ENV HUGGINGFACE_HUB_CACHE=/tmp \
    PORT=8080

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Google CLI single command
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# COPY custom entrypoint for Google
COPY --chmod=775 containers/tei/cpu/1.5.1/entrypoint.sh entrypoint.sh

FROM base as grpc

COPY --from=grpc-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]

FROM base AS http

COPY --from=http-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]
