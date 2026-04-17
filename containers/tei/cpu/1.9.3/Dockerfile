FROM alpine AS text-embeddings-inference

RUN mkdir -p /text-embeddings-inference
ADD https://github.com/huggingface/text-embeddings-inference/archive/refs/tags/v1.9.3.tar.gz /text-embeddings-inference/sources.tar.gz
RUN tar -C /text-embeddings-inference -xf /text-embeddings-inference/sources.tar.gz --strip-components=1

FROM lukemathwalker/cargo-chef:latest-rust-1.92-bookworm AS chef
WORKDIR /usr/src

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel=2024.0.0-49656 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" > fakeintel.c && \
    gcc -shared -fPIC -o libfakeintel.so fakeintel.c

COPY --from=planner /usr/src/recipe.json recipe.json

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo chef cook --release --features ort,candle,mkl,static-linking,http,google --no-default-features --recipe-path recipe.json && sccache -s

COPY --from=text-embeddings-inference /text-embeddings-inference/backends backends
COPY --from=text-embeddings-inference /text-embeddings-inference/core core
COPY --from=text-embeddings-inference /text-embeddings-inference/router router
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.toml Cargo.toml
COPY --from=text-embeddings-inference /text-embeddings-inference/Cargo.lock Cargo.lock

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo build --release --bin text-embeddings-router --features ort,candle,mkl,static-linking,http,google --no-default-features && sccache -s

FROM debian:bookworm-slim AS base

ENV HUGGINGFACE_HUB_CACHE=/tmp \
    PORT=8080 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=4 \
    LD_PRELOAD=/usr/local/libfakeintel.so \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    libomp-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_def.so.2 /usr/local/lib/libmkl_vml_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2 /usr/local/lib/libmkl_def.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx2.so.2 /usr/local/lib/libmkl_vml_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_vml_avx512.so.2 /usr/local/lib/libmkl_vml_avx512.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2 /usr/local/lib/libmkl_avx2.so.2
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx512.so.2 /usr/local/lib/libmkl_avx512.so.2
COPY --from=builder /usr/src/libfakeintel.so /usr/local/libfakeintel.so

COPY --from=builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

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

COPY --chmod=775 containers/tei/cpu/1.9.3/entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]
