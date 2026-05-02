# 🤗 Hugging Face on Google Cloud

Hugging Face collaborates with Google Cloud across open science, open source, and cloud; to enable companies and individuals to build their own AI with the latest open models from Hugging Face and the latest features and innovation from Google Cloud.

![Hugging Face on Google Cloud](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/google-cloud/thumbnail.png)

This repository contains the [Hugging Face Deep Learning Containers (DLCs) for Google Cloud](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face), as well as documentation and dedicated examples ranging different scenarios and use-cases, showcasing how to benefit from Hugging Face on Google Cloud, specifically targeting Vertex AI, Google Kubernetes Engine, and Cloud Run.

The containers are publicly maintained, updated and released periodically by Hugging Face and Google Cloud, and made available for all Google Cloud customers on Google Cloud, under the us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden Artifact Registry.

## DLCs

The DLCs are a set of optimized, pre-installed Docker images designed to streamline the development, training, and deployment of machine learning models on Google Cloud. DLCs provide pre-configured Docker containers for solutions like [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) or PyTorch-based frameworks as [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), or [Sentence Transformers](https://github.com/huggingface/sentence-transformers) for inference, or [TRL](https://github.com/huggingface/trl) for post-training; among others. The DLCs are optimized for performance and efficiency on CPUs, NVIDIA GPUs, and Google TPUs, and come with all the required drivers to work seamlessly on Google Cloud on such devices. Additionally, the DLCs are regularly maintained and released, designed to integrate seamlessly on Google Cloud, and the vulnerabilities are regularly scanned and patched.

> [!NOTE]
> Run the following command to list all the currently available Hugging Face DLCs:
>
> ```bash
> gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden" | grep "hf-" | grep "official"
> ```
>
> Or rather run the following to list all the former but still available Hugging Face DLCs:
>
> ```bash
> gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-"
> ```

### Inference

For inference, at the moment the DLCs cover both:

- Text Embeddings Inference (TEI), which is Hugging Face solution for inference with embedding models written in Rust.

    |                                                                                                Container URI |   Platform |
    |--------------------------------------------------------------------------------------------------------------|------------|
    |         us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/hf-tei-official-cpu.1-9.debian12 |        CPU |
    | us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/hf-tei-official-gpu.cu129.1-9.ubuntu2404 | NVIDIA GPU |

- PyTorch Inference, comes with PyTorch as well as Hugging Face libraries as Transformers, Diffusers, or Sentence Transformers, to expose the task-based APIs for each model on a wide-range of tasks from text-classification on Transformers, text-to-image on Diffusers, or feature-extraction on Sentence Transformers, among many others.

    |                                                                                                                    Container URI |   Platform |
    |----------------------------------------------------------------------------------------------------------------------------------|------------|
    | us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/hf-pytorch-inference-official-gpu.cu128.2-7.ubuntu2404.py311 | NVIDIA GPU |

> [!NOTE]
> The inference DLCs come with native support for Vertex AI, meaning that those will handle the `AIP_...`-like environment variables, as well as expose Vertex AI compliant routes that match its specification.

### Training

For training, the DLCs come with PyTorch as well as Hugging Face libraries as Transformers, Diffusers, Sentence Transformers; but also other libraries as Accelerate or TRL, for distributed training and post-training, respectively; allowing users to easily fine-tune any model on different settings, from text classification models to post-training LLMs and VLMs.

|                                                                                                                    Container URI |  Platform |
|----------------------------------------------------------------------------------------------------------------------------------|-----------|
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-48.ubuntu2204.py311 | NVIDIA GPU |
