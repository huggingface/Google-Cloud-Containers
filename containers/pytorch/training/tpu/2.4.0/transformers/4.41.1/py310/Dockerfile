FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla@sha256:f6504628f3ab57acd4bc0de1fed0b1b8d6a4b71c6f11bc2b08df971b5627162d
# The nightly image with nightly version of PyTorch=2.4, Python=3.10
# Read more about it here: https://github.com/pytorch/xla?tab=readme-ov-file#docker

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG OPTIMUM_TPU='2ec2983'
ARG TRANSFORMERS='4.41.1'
ARG PEFT='0.10.0'
ARG TRL='0.8.6'
ARG DATASETS='2.19.1'
ARG ACCELERATE='0.30.1'
ARG EVALUATE='0.4.2'

# Update pip
RUN pip install --upgrade pip

# Install Hugging Face Libraries
RUN pip install --upgrade --no-cache-dir \
  transformers[sklearn,sentencepiece]==${TRANSFORMERS} \
  datasets==${DATASETS} \
  accelerate==${ACCELERATE} \
  evaluate==${EVALUATE} \
  peft==${PEFT} \
  trl==${TRL} \
  jupyter notebook

# Install Optimum TPU
RUN pip install git+https://github.com/huggingface/optimum-tpu.git@${OPTIMUM_TPU}
# Add example
ADD https://raw.githubusercontent.com/huggingface/optimum-tpu/v0.1.0a0/examples/language-modeling/gemma_tuning.ipynb \
  /notebooks/gemma_tuning.ipynb


# Install Google Cloud Dependencies
RUN pip install --upgrade --no-cache-dir \
  google-cloud-storage \
  google-cloud-bigquery \
  google-cloud-aiplatform \
  google-cloud-pubsub \
  google-cloud-logging \
  "protobuf<4.0.0"

# Override pytorch xla base image with empty entrypoint
ENTRYPOINT [""]
