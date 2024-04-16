# Hugging Face Deep Learning Containers for Google Cloud

This repository contains container files for building Hugging Face specific Deep Learning Containers, examples and tests for Google Cloud.

## Usage Examples

The [`examples`](./examples) directory contains examples for using the containers.

## Building the Containers manually

* [Text Generation Inference](./containers/tgi/README.md)
* [Text Embedding Inference](./containers/tei/README.md)

## Configurations

> Need to be implemented

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of the container. Google uses this file to determine which container to build as the latest version.

## Tests

After the containers are built, you can run the tests in the `tests` directory to verify that they are working correctly.

## Available Containers

| Container Tag                                                                                                                    | Framework | Type      | Accelerator |
| -------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | ----------- |
| [pytorch-training-gpu.2.1.transformers.4.38.1.py310](./containers/pytorch/training/gpu/2.1/transformers/4.38.1/py310/Dockerfile) | Pytorch   | training  | GPU         |
| [text-generation-inference-gpu.2.0.0](./containers/tgi/gpu/2.0.0/Dockerfile)                                                     | -         | inference | GPU         |
| [text-embedding-inference-cpu.1.2.1](./containers/tei/cpu/1.2.1/Dockerfile)                                                      | -         | inference | CPU         |
| [text-embedding-inference-gpu.1.2.1](./containers/tei/gpu/1.2.1/Dockerfile)                                                      | -         | inference | GPU         |

## Directory Structure

The container files are organized in a nested folder structure based on the container tag. For example, the Dockerfile for the container with the tag `pytorch-training-gpu.2.1.transformers.4.38.1.py310` is located at `pytorch/training/gpu/2.1/transformers/4.38.1/py310/Dockerfile`.

## Updates

When we update the transformers version, we add a new folder in the `transformers` directory. For example, if we update the transformers version to 4.39.0, we would add a new folder at `pytorch/training/gpu/2.0/transformers/4.39.0`.
