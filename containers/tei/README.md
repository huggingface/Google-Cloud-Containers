# Text-Embeddings-Inference

[Text-Embeddings-Inference](https://github.com/huggingface/text-embeddings-inference) is a blazing fast inference solution for text embeddings models. You can use any JinaBERT model with Alibi or absolute positions or any BERT, CamemBERT, RoBERTa, or XLM-RoBERTa model with absolute positions in text-embeddings-inference.

## Getting Started

Below are the instructions to build and test the Text-Embedding-Inference container.

### CPU

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embedding-inference-cpu.1.4.0 -f containers/tei/cpu/1.4.0/Dockerfile .
```

Simulate Vertex AI environment with the following command:

```bash
model=BAAI/bge-large-en-v1.5
AIP_PREDICT_ROUTE=/t
AIP_HEALTH_ROUTE=/h
AIP_HTTP_PORT=8080
docker run -ti -p 8080:8080 -e MODEL_ID=$model -e AIP_PREDICT_ROUTE=$AIP_PREDICT_ROUTE -e AIP_HEALTH_ROUTE=$AIP_HEALTH_ROUTE -e AIP_HTTP_PORT=$AIP_HTTP_PORT us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embedding-inference-cpu.1.4.0
```

Send request:

```bash
curl 127.0.0.1:8080/t \
    -X POST \
    -d '{"instances":[{"inputs":"Deep Learning is a"}]}' \
    -H 'Content-Type: application/json'
```

### GPU

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embedding-inference-gpu.1.4.0 -f containers/tei/gpu/1.4.0/Dockerfile .
```

Simulate GKE environment with the following command:

```bash
model=BAAI/bge-large-en-v1.5
docker run -ti --gpus all -p 8080:8080 -e MODEL_ID=$model us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-embedding-inference-gpu.1.4.0
```

Send request:

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"Deep Learning is a"}' \
    -H 'Content-Type: application/json'
```
