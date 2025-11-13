# 🤗 Hugging Face Deep Learning Containers (DLCs) for Google Cloud

<img alt="Hugging Face x Google Cloud" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/google-cloud/thumbnail.png" />

[Hugging Face Deep Learning Containers (DLCs) for Google Cloud](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face) are a set of Docker images for training and deploying Transformers, Sentence Transformers, and Diffusers models on Google Cloud Vertex AI, Google Kubernetes Engine (GKE), and Google Cloud Run.

The [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers/tree/main) repository contains the container files for building Hugging Face-specific Deep Learning Containers (DLCs), examples on how to train and deploy models on Google Cloud.

The containers are publicly maintained, updated and released periodically by Hugging Face and the Google Cloud and available for all Google Cloud customers in the [Google Cloud's Artifact Registry](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face).

- Training
  - [PyTorch](./containers/pytorch/training/README.md)
    - GPU
    - TPU (soon to be released)
- Inference
  - [PyTorch](./containers/pytorch/inference/README.md)
    - CPU
    - GPU
  - [Text Generation Inference](./containers/tgi/README.md)
    - GPU
    - TPU (soon to be released)
  - [Text Embeddings Inference](./containers/tei/README.md)
    - CPU
    - GPU

## Latest DLCs

| Container URI                                                                                                                     | Path                                                                                                                                               | Framework | Type      | Accelerator |
| --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | ----------- |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311           | [text-generation-inference-gpu.2.4.0](./containers/tgi/gpu/2.4.0/Dockerfile)                                                                       | TGI       | Inference | GPU         |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-6.ubuntu2204                 | [text-embeddings-inference-gpu.1.6.0](./containers/tei/gpu/1.6.0/Dockerfile)                                                                       | TEI       | Inference | GPU         |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cpu.1-6                              | [text-embeddings-inference-cpu.1.6.0](./containers/tei/cpu/1.6.0/Dockerfile)                                                                       | TEI       | Inference | CPU         |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310  | [huggingface-pytorch-training-gpu.2.3.1.transformers.4.48.0.py311](./containers/pytorch/training/gpu/2.3.1/transformers/4.48.0/py311/Dockerfile)   | PyTorch   | Training  | GPU         |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-48.ubuntu2204.py311 | [huggingface-pytorch-inference-gpu.2.3.1.transformers.4.48.0.py311](./containers/pytorch/inference/gpu/2.3.1/transformers/4.48.0/py311/Dockerfile) | PyTorch   | Inference | GPU         |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-3.transformers.4-48.ubuntu2204.py311   | [huggingface-pytorch-inference-cpu.2.3.1.transformers.4.48.0.py311](./containers/pytorch/inference/cpu/2.3.1/transformers/4.48.0/py311/Dockerfile) | PyTorch   | Inference | CPU         |

> [!NOTE]
> The listing above **only contains the latest version of each of the Hugging Face DLCs**, the full listing of the available published containers in Google Cloud can be found either in the [Deep Learning Containers Documentation](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face), in the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or via the `gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-"` command.

## Examples

The [`examples`](./examples) directory contains examples for using the containers on different scenarios, and digging deeper on some of the features of the containers offered within Google Cloud.

### Training

| Service   | Example                                                                                                                                    | Title                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Vertex AI | [examples/vertex-ai/notebooks/trl-lora-sft-fine-tuning-on-vertex-ai](./examples/vertex-ai/notebooks/trl-lora-sft-fine-tuning-on-vertex-ai) | Fine-tune Gemma 2B with PyTorch Training DLC using SFT + LoRA on Vertex AI  |
| Vertex AI | [examples/vertex-ai/notebooks/trl-full-sft-fine-tuning-on-vertex-ai](./examples/vertex-ai/notebooks/trl-full-sft-fine-tuning-on-vertex-ai) | Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT on Vertex AI  |
| GKE       | [examples/gke/trl-full-fine-tuning](./examples/gke/trl-full-fine-tuning)                                                                   | Fine-tune Gemma 2B with PyTorch Training DLC using SFT on GKE               |
| GKE       | [examples/gke/trl-lora-fine-tuning](./examples/gke/trl-lora-fine-tuning)                                                                   | Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT + LoRA on GKE |

### Inference

| Service   | Example                                                                                                                              | Title                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-bert-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-bert-on-vertex-ai)                     | Deploy BERT Models with PyTorch Inference DLC on Vertex AI    |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-embedding-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-embedding-on-vertex-ai)           | Deploy Embedding Models with TEI DLC on Vertex AI             |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-flux-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-flux-on-vertex-ai)                     | Deploy FLUX with PyTorch Inference DLC on Vertex AI           |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-gemma-from-gcs-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-gemma-from-gcs-on-vertex-ai) | Deploy Gemma 7B with TGI DLC from GCS on Vertex AI            |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai)                   | Deploy Gemma 7B with TGI DLC on Vertex AI                     |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-llama-vision-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-llama-vision-on-vertex-ai)     | Deploy Llama 3.2 11B Vision with TGI DLC on Vertex AI         |
| Vertex AI | [examples/vertex-ai/notebooks/deploy-llama-3-1-405b-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-llama-3-1-405b-on-vertex-ai) | Deploy Meta Llama 3.1 405B with TGI DLC on Vertex AI          |
| GKE       | [examples/gke/tei-from-gcs-deployment](./examples/gke/tei-from-gcs-deployment)                                                       | Deploy BGE Base v1.5 with TEI DLC from GCS on GKE             |
| GKE       | [examples/gke/tgi-multi-lora-deployment](./examples/gke/tgi-multi-lora-deployment)                                                   | Deploy Gemma2 with multiple LoRA adapters with TGI DLC on GKE |
| GKE       | [examples/gke/tgi-llama-405b-deployment](./examples/gke/tgi-llama-405b-deployment)                                                   | Deploy Llama 3.1 405B with TGI DLC on GKE                     |
| GKE       | [examples/gke/tgi-llama-vision-deployment](./examples/gke/tgi-llama-vision-deployment)                                               | Deploy Llama 3.2 11B Vision with TGI DLC on GKE               |
| GKE       | [examples/gke/tgi-deployment](./examples/gke/tgi-deployment)                                                                         | Deploy Meta Llama 3 8B with TGI DLC on GKE                    |
| GKE       | [examples/gke/tgi-from-gcs-deployment](./examples/gke/tgi-from-gcs-deployment)                                                       | Deploy Qwen2 7B with TGI DLC from GCS on GKE                  |
| GKE       | [examples/gke/tei-deployment](./examples/gke/tei-deployment)                                                                         | Deploy Snowflake's Arctic Embed with TEI DLC on GKE           |
| Cloud Run | [examples/cloud-run/deploy-gemma-2-on-cloud-run](./examples/cloud-run/deploy-gemma-2-on-cloud-run)                                   | Deploy Gemma2 9B with TGI DLC on Cloud Run                    |
| Cloud Run | [examples/cloud-run/deploy-llama-3-1-on-cloud-run](./examples/cloud-run/deploy-llama-3-1-on-cloud-run)                               | Deploy Llama 3.1 8B with TGI DLC on Cloud Run                 |

### Evaluation

| Service   | Example                                                                                                                  | Title                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| Vertex AI | [examples/vertex-ai/notebooks/evaluate-llms-with-vertex-ai](./examples/vertex-ai/notebooks/evaluate-llms-with-vertex-ai) | Evaluate open LLMs with Vertex AI and Gemini |
