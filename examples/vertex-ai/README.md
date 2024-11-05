# Vertex AI Examples

This directory contains usage examples of the Hugging Face Deep Learning Containers (DLCs) in Google Vertex AI for both training and inference, with a focus on Large Language Models (LLMs), while also including some examples showcasing how to train and deploy models suited for other task than text generation.

For Google Vertex AI, we differentiate between the executable Jupyter Notebook examples, which are located in the [notebooks](./notebooks) directory, and the Kubeflow examples, which are located in the [pipelines](./pipelines) directory.

## Notebooks

### Training Examples

| Example                                                                                              | Title                                                                      |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [notebooks/trl-lora-sft-fine-tuning-on-vertex-ai](./notebooks/trl-lora-sft-fine-tuning-on-vertex-ai) | Fine-tune Gemma 2B with PyTorch Training DLC using SFT + LoRA on Vertex AI |
| [notebooks/trl-full-sft-fine-tuning-on-vertex-ai](./notebooks/trl-full-sft-fine-tuning-on-vertex-ai) | Fine-tune Mistral 7B v0.3 with PyTorch Training DLC using SFT on Vertex AI |

### Inference Examples

| Example                                                                                        | Title                                                      |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [notebooks/deploy-bert-on-vertex-ai](./notebooks/deploy-bert-on-vertex-ai)                     | Deploy BERT Models with PyTorch Inference DLC on Vertex AI |
| [notebooks/deploy-embedding-on-vertex-ai](./notebooks/deploy-embedding-on-vertex-ai)           | Deploy Embedding Models with TEI DLC on Vertex AI          |
| [notebooks/deploy-flux-on-vertex-ai](./notebooks/deploy-flux-on-vertex-ai)                     | Deploy FLUX with PyTorch Inference DLC on Vertex AI        |
| [notebooks/deploy-gemma-from-gcs-on-vertex-ai](./notebooks/deploy-gemma-from-gcs-on-vertex-ai) | Deploy Gemma 7B with TGI DLC from GCS on Vertex AI         |
| [notebooks/deploy-gemma-on-vertex-ai](./notebooks/deploy-gemma-on-vertex-ai)                   | Deploy Gemma 7B with TGI DLC on Vertex AI                  |
| [notebooks/deploy-llama-vision-on-vertex-ai](./notebooks/deploy-llama-vision-on-vertex-ai)     | Deploy Llama 3.2 11B Vision with TGI DLC on Vertex AI      |
| [notebooks/deploy-llama-3-1-405b-on-vertex-ai](./notebooks/deploy-llama-3-1-405b-on-vertex-ai) | Deploy Meta Llama 3.1 405B with TGI DLC on Vertex AI       |

### Evaluation Examples

| Example                                                                            | Title                                        |
| ---------------------------------------------------------------------------------- | -------------------------------------------- |
| [notebooks/evaluate-llms-with-vertex-ai](./notebooks/evaluate-llms-with-vertex-ai) | Evaluate open LLMs with Vertex AI and Gemini |

## Pipelines

Coming soon!