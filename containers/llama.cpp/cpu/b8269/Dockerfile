ARG UBUNTU_VERSION=24.04

FROM alpine AS llama-cpp

RUN mkdir -p /llama.cpp
ADD https://github.com/ggerganov/llama.cpp/archive/refs/tags/b8269.tar.gz /llama.cpp/sources.tar.gz
RUN tar -C /llama.cpp -xf /llama.cpp/sources.tar.gz --strip-components=1

FROM ubuntu:$UBUNTU_VERSION AS build

ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y build-essential git cmake libssl-dev

WORKDIR /app

COPY --from=llama-cpp /llama.cpp /app

RUN if [ "$TARGETARCH" = "amd64" ] || [ "$TARGETARCH" = "arm64" ]; then \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON; \
    else \
    echo "Unsupported architecture"; \
    exit 1; \
    fi && \
    cmake --build build -j $(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ubuntu:$UBUNTU_VERSION AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl gnupg apt-transport-https ca-certificates \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

COPY --from=build /app/lib/ /app

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0 \
    HF_HOME=/tmp

COPY --from=build /app/full/llama-server /app
COPY --chmod=775 containers/llama.cpp/cpu/b8269/entrypoint.sh /app/entrypoint.sh

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/entrypoint.sh" ]
