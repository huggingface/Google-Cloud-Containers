# Google Kubernetes Engine (GKE) Examples

This directory contains usage examples of the Hugging Face Deep Learning Containers (DLCs) in Google Kubernetes Engine (GKE) for both training and inference, with a focus on Large Language Models (LLMs).

## Training Examples

| Example | Description
|---------|-------------
| [trl-full-fine-tuning](./trl-full-fine-tuning) | Full SFT fine-tuning of Gemma 2B in a multi-GPU instance with TRL.
| [trl-lora-fine-tuning](./trl-lora-fine-tuning) | LoRA SFT fine-tuning of Mistral 7B v0.3 in a single GPU instance with TRL.

## Inference Examples

| Example | Description
|---------|-------------
| [tgi-deployment](./tgi-deployment) | Deploying Llama3 8B with Text Generation Inference (TGI) in GKE.
| [tgi-from-gcs-deployment](./tgi-from-gcs-deployment) | Deploying Qwen2 7B Instruct with Text Generation Inference (TGI) from a GCS Bucket in GKE.
| [tei-deployment](./tei-deployment) | Deploying Snowflake's Artic Embed (M) with Text Embeddings Inference (TEI) in GKE.
| [tei-from-gcs-deployment](./tei-from-gcs-deployment) | Deploying BGE Base v1.5 (English) with Text Embeddings Inference (TEI) from a GCS Bucket in GKE.
