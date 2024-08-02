# 🤗 Deep Learning Containers (DLCs)

This directory contains the Hugging Face Deep Learning Containers (DLCs) available for Google Cloud, divided by framework (PyTorch, TGI or TEI), with the difference that PyTorch covers both training and inference, while TGI and TEI are only for inference.

Additionally, the [`container.yaml`](./container.yaml) file contains the configuration for the latest version of each container. Google uses this file to determine which container to build as of the latest version, but can also be used as a reference on the latest available containers.

Find all the available Hugging Face DLCs in either [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or list those using the following `gcloud` command that lists the containers with the tag containing `huggingface-` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface"
```
