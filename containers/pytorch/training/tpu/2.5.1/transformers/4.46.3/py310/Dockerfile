FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.1_3.10_tpuvm
# Google maintained pytorch xla image with PyTorch=2.5.1 and Python=3.10
# Read more about it here: https://github.com/pytorch/xla?tab=readme-ov-file#docker

LABEL maintainer="Hugging Face"
ARG DEBIAN_FRONTEND=noninteractive

# Versions
ARG OPTIMUM_TPU='0.2.2'
ARG TRANSFORMERS='4.46.3'
ARG PEFT='0.13.2'
ARG TRL='0.12.1'
ARG DATASETS='3.1.0'
ARG ACCELERATE='1.1.0'
ARG EVALUATE='0.4.3'
ARG SAFETENSORS='0.4.5'

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
  safetensors==${SAFETENSORS} \
  jupyter notebook

# Install Optimum TPU
RUN pip install git+https://github.com/huggingface/optimum-tpu.git@v${OPTIMUM_TPU}
# Add example
ADD https://raw.githubusercontent.com/huggingface/optimum-tpu/v${OPTIMUM_TPU}/examples/language-modeling/gemma_tuning.ipynb \
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
