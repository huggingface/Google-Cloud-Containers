# 🤗 Deep Learning Containers (DLCs)

This directory contains the Hugging Face Deep Learning Containers (DLCs) available for Google Cloud, divided by framework (PyTorch, TGI or TEI), with the difference that PyTorch covers both training and inference, while TGI and TEI are only for inference.

Additionally, the [`container.yaml`](./container.yaml) file contains the configuration for the latest version of each container. Google uses this file to determine which container to build as of the latest version, but can also be used as a reference on the latest available containers.
