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
