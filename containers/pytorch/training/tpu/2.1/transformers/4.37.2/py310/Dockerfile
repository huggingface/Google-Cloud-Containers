FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm
# The image with PyTorch = 2.1, Python=3.10
# Read more about it here: https://github.com/pytorch/xla?tab=readme-ov-file#docker

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG TRANSFORMERS='4.37.2'
ARG DIFFUSERS='0.26.1'
ARG PEFT='0.8.2'
ARG TRL='0.7.10'
ARG DATASETS='2.16.1'
ARG ACCELERATE='0.27.0'
ARG EVALUATE='0.4.1'
ARG SENTENCE_TRANSFORMERS='2.3.1'
ARG NOTEBOOK='7.1.0'

RUN apt-get update \
    && apt-get install -y \
    bzip2 \
    curl \
    git \
    git-lfs \
    tar \
    gcc \
    g++ \
    libaio-dev \
    # audio
    libsndfile1-dev \
    ffmpeg \
    apt-transport-https \
    gnupg \
    ca-certificates \
    && apt-get clean autoremove --yes 

# Update pip
RUN pip install --upgrade pip

# Install Hugging Face Libraries
RUN pip install --upgrade --no-cache-dir \
  transformers[sklearn,sentencepiece,vision]==${TRANSFORMERS} \
  diffusers==${DIFFUSERS} \
  datasets==${DATASETS} \
  accelerate==${ACCELERATE} \
  evaluate==${EVALUATE} \
  peft==${PEFT} \
  trl==${TRL} \
  sentence-transformers==${SENTENCE_TRANSFORMERS} \
  notebook==${NOTEBOOK}

#Install Google Cloud Dependencies
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

RUN pip install --upgrade --no-cache-dir \
  google-cloud-storage \ 
  google-cloud-bigquery \
  google-cloud-aiplatform \
  google-cloud-pubsub \
  google-cloud-logging

# Check if correct versions are installed
RUN python -c "import transformers, diffusers, datasets, accelerate, evaluate, peft, trl, sentence_transformers, torch; \
                assert all([mod.__version__ == version for mod, version in [(transformers, '${TRANSFORMERS}'), (diffusers, '${DIFFUSERS}'), \
                (datasets, '${DATASETS}'), (accelerate, '${ACCELERATE}'), (evaluate, '${EVALUATE}'), (peft, '${PEFT}'), (trl, '${TRL}'), \
                (sentence_transformers, '${SENTENCE_TRANSFORMERS}'), (torch, '2.1.0')]])" 