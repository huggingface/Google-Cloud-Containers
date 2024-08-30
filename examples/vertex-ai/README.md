# Vertex AI Examples

This directory contains usage examples of the Hugging Face Deep Learning Containers (DLCs) in Google Vertex AI for both training and inference, with a focus on Large Language Models (LLMs), while also including some examples showcasing how to train and deploy models suited for other task than text generation.

For Google Vertex AI, we differentiate between the executable Jupyter Notebook examples, which are located in the [notebooks](./notebooks) directory, and the Kubeflow examples, which are located in the [pipelines](./pipelines) directory.

## Notebooks

### Training Examples

| Example                                                                                    | Description                                                                             |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| [trl-full-sft-fine-tuning-on-vertex-ai](./notebooks/trl-full-sft-fine-tuning-on-vertex-ai) | Full SFT fine-tuning of Mistral 7B v0.3 in a multi-GPU instance with TRL on Vertex AI.  |
| [trl-lora-sft-fine-tuning-on-vertex-ai](./notebooks/trl-lora-sft-fine-tuning-on-vertex-ai) | LoRA SFT fine-tuning of Mistral 7B v0.3 in a single GPU instance with TRL on Vertex AI. |

### Inference Examples

| Example                                                                              | Description                                                                                                                                     |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [deploy-bert-on-vertex-ai](./notebooks/deploy-bert-on-vertex-ai)                     | Deploying a BERT model for a text classification task using `huggingface-inference-toolkit` for a Custom Prediction Routine (CPR) on Vertex AI. |
| [deploy-embedding-on-vertex-ai](./notebooks/deploy-embedding-on-vertex-ai)           | Deploying an embedding model with Text Embeddings Inference (TEI) on Vertex AI.                                                                 |
| [deploy-gemma-on-vertex-ai](./notebooks/deploy-gemma-on-vertex-ai)                   | Deploying Gemma 7B Instruct with Text Generation Inference (TGI) on Vertex AI.                                                                  |
| [deploy-gemma-from-gcs-on-vertex-ai](./notebooks/deploy-gemma-from-gcs-on-vertex-ai) | Deploying Gemma 7B Instruct with Text Generation Inference (TGI) from a GCS Bucket on Vertex AI.                                                |

## Pipelines

More to come soon!
