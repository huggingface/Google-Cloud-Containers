# Google Kubernetes Engine (GKE) Examples

This directory contains usage examples of the Hugging Face Deep Learning Containers (DLCs) in Google Kubernetes Engine (GKE) for both training and inference, with a focus on Large Language Models (LLMs).

## Training Examples

| Example                                        | Title                                                                       |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| [trl-full-fine-tuning](./trl-full-fine-tuning) | Fine-tune Gemma 2B with PyTorch Training DLC using SFT on GKE               |
| [trl-lora-fine-tuning](./trl-lora-fine-tuning) | Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT + LoRA on GKE |

## Inference Examples

| Example                                                      | Title                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| [tei-deployment](./tei-deployment)                           | Deploy Snowflake's Arctic Embed with TEI DLC on GKE           |
| [tei-from-gcs-deployment](./tei-from-gcs-deployment)         | Deploy BGE Base v1.5 with TEI DLC from GCS on GKE             |
| [tgi-deployment](./tgi-deployment)                           | Deploy Meta Llama 3 8B with TGI DLC on GKE                    |
| [tgi-from-gcs-deployment](./tgi-from-gcs-deployment)         | Deploy Qwen2 7B with TGI DLC from GCS on GKE                  |
| [tgi-llama-405b-deployment](./tgi-llama-405b-deployment)     | Deploy Llama 3.1 405B with TGI DLC on GKE                     |
| [tgi-llama-vision-deployment](./tgi-llama-vision-deployment) | Deploy Llama 3.2 11B Vision with TGI DLC on GKE               |
| [tgi-multi-lora-deployment](./tgi-multi-lora-deployment)     | Deploy Gemma2 with multiple LoRA adapters with TGI DLC on GKE |
