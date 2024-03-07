# Hugging Face Deep Learning Containers for Google Cloud

This repository contains container files for building Hugging Face specific Deep Learning Containers, examples and tests for Google Cloud.

## Usage Examples

The [`examples`](./examples) directory contains examples for using the containers.

## Building the Containers without (container.yaml)

_Note: we added the latest TGI version as an example into the repository, which can be build with._

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.4.2 -f containers/tgi/gpu/1.4.2/Dockerfile .
```

### Mistral 7B test

test the container on a GPU instance (g2) with

```bash
model=mistralai/Mistral-7B-Instruct-v0.2
num_shard=1
max_input_length=1562
max_total_tokens=2048

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.4.2  
```

Send request:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] What is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```

### Gemma Test

```bash
model=google/gemma-7b
num_shard=1
max_input_length=512
max_total_tokens=1024
max_batch_prefill_tokens=1512
token=YOUR_TOKEN

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -e MAX_BATCH_PREFILL_TOKENS=$max_batch_prefill_tokens \
  -e HUGGING_FACE_HUB_TOKEN=$token \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.4.2
```

Send request:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Deep Learning is a","parameters":{"temperature":1.0, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```

For a Vertex AI example checkout [Deploy Gemma on Vertex AI](./examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai.ipynb)


## Configurations

> Need to be implemented

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of the container. Google uses this file to determine which container to build as the latest version.

## Tests

After the containers are built, you can run the tests in the `tests` directory to verify that they are working correctly.

## Available Containers

| Container Tag                                                                 | Framework | Type      | Accelerator |
| ----------------------------------------------------------------------------- | --------- | --------- | ----------- |
| [pytorch-training-gpu.2.1.transformers.4.38.1.py310](./containers/pytorch/training/gpu/2.1/transformers/4.38.1/py310/Dockerfile)  | Pytorch   | training  | GPU         |
| [text-generation-inference-gpu.1.4.2](./containers/tgi/gpu/1.4.2/Dockerfile)                 | -         | inference | GPU         |

## Directory Structure

The container files are organized in a nested folder structure based on the container tag. For example, the Dockerfile for the container with the tag `pytorch-training-gpu.2.1.transformers.4.38.1.py310` is located at `pytorch/training/gpu/2.1/transformers/4.38.1/py310/Dockerfile`.

## Updates

When we update the transformers version, we add a new folder in the `transformers` directory. For example, if we update the transformers version to 4.39.0, we would add a new folder at `pytorch/training/gpu/2.0/transformers/4.39.0`.
