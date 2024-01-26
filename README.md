# Hugging Face Deep Learning Containers for Google Cloud

This repository contains container files for building Hugging Face specific Deep Learning Containers, examples and tests for Google Cloud.

## Examples 

The [`examples`](./examples) directory contains examples for using the containers. 


## Building the Containers without (container.yaml)

_Note: we added the latest TGI version as an example into the repository, which can be build with._

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.3.4 -f containers/tgi/gpu/1.3.4/Dockerfile .
```

### Mistral 7B test

test the container on a GPU instance with

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
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.3.4  
```

Send request:
``` 
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] What is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```

### Golden Gate Test

```bash
model=gg-hf/golden-gate-7b
num_shard=1
max_input_length=1562
max_total_tokens=2048
max_batch_prefill_tokens=3000
token=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

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -e MAX_BATCH_PREFILL_TOKENS=$max_batch_prefill_tokens \
  -e HUGGING_FACE_HUB_TOKEN=$token \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.3.4
```

Send request:
``` 
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Deep Learning is","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```


## Configurations

> Need to be implemented

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of the container. Google uses this file to determine which container to build as the latest version. 

## Tests

After the containers are built, you can run the tests in the `tests` directory to verify that they are working correctly.

## Available Containers 

> Placeholder for the table

| Container Tag                                                                 | Framework | Type      | Accelerator |
| ----------------------------------------------------------------------------- | --------- | --------- | ----------- |
| [pytorch-training-gpu.2.0.transformers.4.35.0.py310](link_to_container_here)  | Pytorch   | training  | GPU         |
| [pytorch-training-tpu.2.0.transformers.4.35.0.py310](link_to_container_here)  | Pytorch   | training  | TPU         |
| [pytorch-inference-gpu.2.0.transformers.4.35.0.py310](link_to_container_here) | Pytorch   | inference | GPU         |
| [pytorch-inference-tpu.2.0.transformers.4.35.0.py310](link_to_container_here) | Pytorch   | inference | TPU         |
| [pytorch-inference-cpu.2.0.transformers.4.35.0.py310](link_to_container_here) | Pytorch   | inference | CPU         |
| [jax-training-gpu.2.0.transformers.4.35.0.py310](link_to_container_here)      | Jax       | training  | GPU         |
| [jax-training-tpu.2.0.transformers.4.35.0.py310](link_to_container_here)      | Jax       | training  | TPU         |
| [jax-inference-gpu.2.0.transformers.4.35.0.py310](link_to_container_here)     | Jax       | inference | GPU         |
| [jax-inference-tpu.2.0.transformers.4.35.0.py310](link_to_container_here)     | Jax       | inference | TPU         |
| [text-generation-inference-gpu.1.3.4](link_to_container_here)                 | -         | inference | GPU         |
| [text-generation-inference-tpu.1.3.4](link_to_container_here)                 | -         | inference | TPU         |
| [text-embedding-inference-gpu.0.6.0](link_to_container_here)                  | -         | inference | GPU         |
| [text-embedding-inference-cpu.0.6.0](link_to_container_here)                  | -         | inference | CPU         |

## Directory Structure

The container files are organized in a nested folder structure based on the container tag. For example, the Dockerfile for the container with the tag `pytorch-training-gpu.2.0.transformers.4.35.0.py310` is located at `pytorch/training/gpu/2.0/transformers/4.35.0/py310/Dockerfile`.



## Updates

When we update the transformers version, we add a new folder in the `transformers` directory. For example, if we update the transformers version to 4.36.0, we would add a new folder at `pytorch/training/gpu/2.0/transformers/4.36.0`.