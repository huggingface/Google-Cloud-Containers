# 🤗 Hugging Face Deep Learning Containers for Google Cloud

This repository contains container files for building Hugging Face specific Deep Learning Containers (DLCs), examples on how to use those for both inference and training for Google Cloud. Ideally this containers are intended to be used within the Google Cloud's Artifact Registry, and released periodically for each supported combination of use-case (training, inference), accelerator type (CPU, GPU, TPU), and framework (PyTorch, TGI, TEI).

* Training
  * [PyTorch](./containers/pytorch/training/README.md)
    * GPU
    * TPU
* Inference
  * [PyTorch](./containers/pytorch/inference/README.md)
    * CPU
    * GPU
    * TPU (soon)
  * [Text Generation Inference](./containers/tgi/README.md)
    * GPU
    * TPU
  * [Text Embeddings Inference](./containers/tei/README.md)
    * CPU
    * GPU

## Usage Examples

The [`examples`](./examples) directory contains examples for using the containers on different scenarios, and digging deeper on some of the features of the containers offered within Google Cloud.

## Available Containers

| Container Tag | Framework | Type | Accelerator |
| --- | --- | --- | --- |
| [text-generation-inference-gpu.2.2.0](./containers/tgi/gpu/2.2.0/Dockerfile) | TGI | Inference | GPU |
| [text-embeddings-inference-gpu.1.4.0](./containers/tei/gpu/1.4.0/Dockerfile) | TEI | Inference | GPU |
| [text-embeddings-inference-cpu.1.4.0](./containers/tei/cpu/1.4.0/Dockerfile) | TEI | Inference | CPU |
| [huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310](./containers/pytorch/training/tpu/2.4.0/transformers/4.41.1/py310/Dockerfile) | PyTorch | Training | TPU |
| [huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310](./containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile) | PyTorch | Training | GPU |
| [huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | GPU |
| [huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | CPU |

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of each container. Google uses this file to determine which container to build as of the latest version, but can also be used as a reference on the latest available containers.

Alternatively, the available i.e. published, Hugging Face DLCs within Google Cloud can be found via `gcloud` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface"
```

## Local development

For documentation on how to create, build and run the containers manually, please navigate to each specific container type directory within this repository:

* [Text Generation Inference](./containers/tgi/README.md)
* [Text Embeddings Inference](./containers/tei/README.md)
* [PyTorch Inference](./containers/pytorch/inference/README.md)
* [PyTorch Training](./containers/pytorch/training/README.md)

## Directory Structure

The container files are organized in a nested folder structure based on the identifiers that define the container tag, making the discoverability of the containers easier, while keeping a 1:1 match with the tags defined for those containers.

For example, if you want to have a look at the Dockerfile for the container with the tag `huggingface-pytorch-training-gpu.2.3.0.transformers.4.42.3.py310` i.e. a container that comes with Python 3.10, with the required NVIDIA Drivers installed in order to enable the GPU usage, and with libraries from the Hugging Face stack useful for training a wide range of models, being `transformers` the main library for that; you should navigate to [`./containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile`](./containers/pytorch/training/gpu/2.3.0/transformers/4.42.0/py310/Dockerfile).

## Updates

When there is a new release of any of the frameworks (`transformers`, `text-generation-inference`, or `text-embeddings-inference`) as well as any other dependency installed within those containers that needs an update or a patch fix, we update the `Dockerfile`; creating a new directory within the [`./containers`](./containers/) directory where applicable and respecting the directory structure mentioned above, and adding the updated `Dockerfile` via a PR to the `main` branch, describing the changes applied.
