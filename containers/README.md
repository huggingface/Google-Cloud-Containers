# 🤗 Deep Learning Containers (DLCs)

This directory contains the Hugging Face Deep Learning Containers (DLCs) available for Google Cloud, divided by framework (PyTorch, TGI or TEI), with the difference that PyTorch covers both training and inference, while TGI and TEI are only for inference.

Additionally, the [`container.yaml`](./container.yaml) file contains the configuration for the latest version of each container. Google uses this file to determine which container to build as of the latest version, but can also be used as a reference on the latest available containers.

Find all the available Hugging Face DLCs in either [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or list those using the following `gcloud` command that lists the containers with the tag containing `huggingface-` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface"
```

The [`container.yaml`](./containers/container.yaml) file contains the configuration for the latest version of each container. Google uses this file to determine which container to build as of the latest version, but can also be used as a reference on the latest available containers.

Alternatively, you can also list the publicly available Hugging Face DLCs in Google Cloud at the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or via `gcloud` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface"
```

## Local development

For documentation on how to build, run, and test the containers manually, please navigate to each specific container type directory within this repository:

* [Text Generation Inference](./containers/tgi/README.md)
* [Text Embeddings Inference](./containers/tei/README.md)
* [PyTorch Inference](./containers/pytorch/inference/README.md)
* [PyTorch Training](./containers/pytorch/training/README.md)

## Directory Structure

The container files are organized in a nested folder structure based on the identifiers that define the container tag, making the discoverability of the containers easier, while keeping a 1:1 match with the tags defined for those containers.

For example, if you want to have a look at the Dockerfile for the container with the tag `huggingface-pytorch-training-gpu.2.3.0.transformers.4.42.3.py310` i.e. a container that comes with Python 3.10, with the required NVIDIA Drivers installed in order to enable the GPU usage, and with libraries from the Hugging Face stack useful for training a wide range of models, being `transformers` the main library for that; you should navigate to [`./containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile`](./containers/pytorch/training/gpu/2.3.0/transformers/4.42.0/py310/Dockerfile).

## Updates

When there is a new release of any of the frameworks (`transformers`, `text-generation-inference`, or `text-embeddings-inference`) as well as any other dependency installed within those containers that needs an update or a patch fix, we update the `Dockerfile`; creating a new directory within the [`./containers`](./containers/) directory where applicable and respecting the directory structure mentioned above, and adding the updated `Dockerfile` via a PR to the `main` branch, describing the changes applied.
