# Hugging Face Deep Learning Containers for Google Cloud

This repository contains Dockerfiles for building Hugging Face specific Deep Learning Containers that are periodically synced and built by Google Cloud.

## Building the Containers without (container.yaml)

_Note: we added the latest TGI version as an example into the repository, which can be build with._

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.3.4 -f tgi/gpu/1.3.4/Dockerfile .
```

## Configurations

> Need to be implemented

The [`container.yaml`](./container.yaml) file contains the configuration for the latest version of the container. Google uses this file to determine which container to build as the latest version. 

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