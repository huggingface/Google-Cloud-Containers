# Hugging Face Deep Learning Containers for Google Cloud

This repository contains container files for building Hugging Face specific Deep Learning Containers, examples and tests for Google Cloud.

We plan to release containers for each of these combinations of framework, use case, and accelerator type:

* Training
  * [Pytorch](./containers/pytorch/training/README.md)
    * GPU
    * TPU
* Inference
  * [Pytorch](./containers/pytorch/inference/README.md)
    * CPU
    * GPU
    * TPU (soon)
  * [Text Generation Inference](./containers/tgi/README.md)
    * GPU
    * TPU (soon)
  * [Text Embeddings Inference](./containers/tei/README.md)
    * CPU
    * GPU

## Usage Examples

The [`examples`](./examples) directory contains examples for using the containers.

## Building the Containers manually

For documentation on how to create the containers manually, please navigate to these README in the `containers` directory.

* [Text Generation Inference](./containers/tgi/README.md)
* [Text Embeddings Inference](./containers/tei/README.md)
* [Pytorch Inference](./containers/pytorch/inference)
* [Pytorch Training](./containers/pytorch/training)

## Configurations

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of the container. Google uses this file to determine which container to build as the latest version.

## Tests

> TODO: Add documentation on how to run the tests for each container (if different).

After the containers are built, you can run the tests in the `tests` directory to verify that they are working correctly.

## Available Containers

| Container Tag | Framework | Type | Accelerator |
| --- | --- | --- | --- |
| [text-generation-inference-gpu.2.1.1](./containers/tgi/gpu/2.1.1/Dockerfile) | TGI | Inference | GPU |
| [text-embeddings-inference-gpu.1.4.0](./containers/tei/gpu/1.4.0/Dockerfile) | TEI | Inference | GPU |
| [text-embeddings-inference-cpu.1.4.0](./containers/tei/cpu/1.4.0/Dockerfile) | TEI | Inference | CPU |
| [huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310](./containers/pytorch/training/tpu/2.4.0/transformers/4.41.1/py310/Dockerfile) | PyTorch | Training | TPU |
| [huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310](./containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile) | PyTorch | Training | GPU |
| [huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | GPU |
| [huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | CPU |

## Directory Structure

The container files are organized in a nested folder structure based on the container tag. For example, if you want to have a look at the  Dockerfile for the container with the tag `huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310`,  navigate to [`/containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile`](./containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile).

## Updates

When we update the transformers version, we add a new folder in the `transformers` directory. For example, if we update the transformers version to 4.39.0, we would add a new folder at `pytorch/training/gpu/2.0/transformers/4.39.0`.
