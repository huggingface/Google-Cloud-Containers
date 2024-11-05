# Examples

This directory contains some usage examples for the Hugging Face Deep Learning Containers (DLCs) available in Google Cloud, as published from the [containers directory](../containers).

The examples' structure is organized based on the Google Cloud service we can use to deploy the containers, being:

- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- [Vertex AI](https://cloud.google.com/vertex-ai)
- (Preview) [Cloud Run](https://cloud.google.com/run)

> [!WARNING]
> Cloud Run now offers on-demand access to NVIDIA L4 GPUs for running AI inference workloads; but is still in preview, so the Cloud Run examples within this repository should be taken solely for testing and experimentation; please avoid using those for production workloads. We are actively working towards general availability and appreciate your understanding.
