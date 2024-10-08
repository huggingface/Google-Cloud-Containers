# (Preview) Cloud Run Examples

This directory contains usage examples of the Hugging Face Deep Learning Containers (DLCs) in Cloud Run only for inference at the moment, with a focus on Large Language Models (LLMs).

> [!WARNING]
> Cloud Run now offers on-demand access to NVIDIA L4 GPUs for running AI inference workloads; but is still in preview, so the Cloud Run examples within this repository should be taken solely for testing and experimentation; please avoid using those for production workloads. We are actively working towards general availability and appreciate your understanding.

## Inference Examples

| Example                            | Title                                           |
| ---------------------------------- | ----------------------------------------------- |
| [tgi-deployment](./tgi-deployment) | Deploy Meta Llama 3.1 with TGI DLC on Cloud Run |
