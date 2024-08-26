# 🤗 Hugging Face Deep Learning Containers for Google Cloud

<img alt="Hugging Face x Google Cloud" src="https://raw.githubusercontent.com/huggingface/blog/main/assets/173_gcp-partnership/thumbnail.jpg" />

[Hugging Face Deep Learning Containers for Google Cloud](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face) are a set of Docker images for training and deploying Transformers, Sentence Transformers, and Diffusers models on Google Cloud Vertex AI and Google Kubernetes Engine (GKE).

The [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers/tree/main) repository contains the container files for building Hugging Face-specific Deep Learning Containers (DLCs), examples on how to train and deploy models on Google Cloud. The containers are publicly maintained, updated and released periodically by Hugging Face and the Google Cloud Team and available for all Google Cloud Customers within the [Google Cloud's Artifact Registry](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face). For each supported combination of use-case (training, inference), accelerator type (CPU, GPU, TPU), and framework (PyTorch, TGI, TEI) containers are created. Those include: 
* Training
  * [PyTorch](./containers/pytorch/training/README.md)
    * GPU
    * TPU (soon)
* Inference
  * [PyTorch](./containers/pytorch/inference/README.md)
    * CPU
    * GPU
    * TPU (soon)
  * [Text Generation Inference](./containers/tgi/README.md)
    * GPU
    * TPU (soon)
  * [Text Embeddings Inference](./containers/tei/README.md)
    * CPU
    * GPU

## Published Containers

| Container URI | Path | Framework | Type | Accelerator |
| --- | --- | --- | --- | --- |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-2.ubuntu2204.py310 | [text-generation-inference-gpu.2.2.0](./containers/tgi/gpu/2.2.0/Dockerfile) | TGI | Inference | GPU |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cu122.1-2.ubuntu2204 | [text-embeddings-inference-gpu.1.2.0](./containers/tei/gpu/1.2.0/Dockerfile) | TEI | Inference | GPU |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embeddings-inference-cpu.1-2 | [text-embeddings-inference-cpu.1.2.0](./containers/tei/cpu/1.2.0/Dockerfile) | TEI | Inference | CPU |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310 | [huggingface-pytorch-training-gpu.2.3.0.transformers.4.42.3.py310](./containers/pytorch/training/gpu/2.3.0/transformers/4.42.3/py310/Dockerfile) | PyTorch | Training | GPU |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-2.transformers.4-41.ubuntu2204.py311 | [huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | GPU |
| us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-2.transformers.4-41.ubuntu2204.py311 | [huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311](./containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile) | PyTorch | Inference | CPU |

> [!NOTE]
> The listing above only contains the latest version of each of the Hugging Face DLCs, the full listing of the available published containers in Google Cloud can be found either in the [Deep Learning Containers Documentation](https://cloud.google.com/deep-learning-containers/docs/choosing-container#hugging-face), in the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or via the `gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-"` command.

## Examples

The [`examples`](./examples) directory contains examples for using the containers on different scenarios, and digging deeper on some of the features of the containers offered within Google Cloud.

### Training Examples

| Service | Example | Description
|---------|---------|-------------
| GKE | [trl-full-fine-tuning](./examples/gke/trl-full-fine-tuning) | Full SFT fine-tuning of Gemma 2B in a multi-GPU instance with TRL on GKE.
| GKE | [trl-lora-fine-tuning](./examples/gke/trl-lora-fine-tuning) | LoRA SFT fine-tuning of Mistral 7B v0.3 in a single GPU instance with TRL on GKE.
| Vertex AI | [trl-full-sft-fine-tuning-on-vertex-ai](./examples/vertex-ai/notebooks/trl-full-sft-fine-tuning-on-vertex-ai) | Full SFT fine-tuning of Mistral 7B v0.3 in a multi-GPU instance with TRL on Vertex AI.
| Vertex AI | [trl-lora-sft-fine-tuning-on-vertex-ai](./examples/vertex-ai/notebooks/trl-lora-sft-fine-tuning-on-vertex-ai) | LoRA SFT fine-tuning of Mistral 7B v0.3 in a single GPU instance with TRL on Vertex AI.

### Inference Examples

| Service | Example | Description
|---------|---------|-------------
| GKE | [tgi-deployment](./examples/gke/tgi-deployment) | Deploying Llama3 8B with Text Generation Inference (TGI) on GKE.
| GKE | [tgi-from-gcs-deployment](./examples/gke/tgi-from-gcs-deployment) | Deploying Qwen2 7B Instruct with Text Generation Inference (TGI) from a GCS Bucket on GKE.
| GKE | [tei-deployment](./examples/gke/tei-deployment) | Deploying Snowflake's Arctic Embed (M) with Text Embeddings Inference (TEI) on GKE.
| GKE | [tei-from-gcs-deployment](./examples/gke/tei-from-gcs-deployment) | Deploying BGE Base v1.5 (English) with Text Embeddings Inference (TEI) from a GCS Bucket on GKE.
| Vertex AI | [deploy-bert-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-bert-on-vertex-ai) | Deploying a BERT model for a text classification task using `huggingface-inference-toolkit` for a Custom Prediction Routine (CPR) on Vertex AI.
| Vertex AI | [deploy-embedding-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-embedding-on-vertex-ai) | Deploying an embedding model with Text Embeddings Inference (TEI) on Vertex AI.
| Vertex AI | [deploy-gemma-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai) | Deploying Gemma 7B Instruct with Text Generation Inference (TGI) on Vertex AI.
| Vertex AI | [deploy-gemma-from-gcs-on-vertex-ai](./examples/vertex-ai/notebooks/deploy-gemma-from-gcs-on-vertex-ai) | Deploying Gemma 7B Instruct with Text Generation Inference (TGI) from a GCS Bucket on Vertex AI.
