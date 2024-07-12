# Text-Generation-Inference

[Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) A Rust, Python and gRPC server for text generation inference. Used in production at Hugging Face to power Hugging Chat, the Inference API and Inference Endpoint.

## Getting Started

Below are the instructions to build and test the Text Generation Inference container.

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.1.1 -f containers/tgi/gpu/2.1.1/Dockerfile .
```

Alternatively, feel free to use the following command to dump the build logs into a file:

```bash
docker build --progress=plain -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.1.1 -f containers/tgi/gpu/2.1.1/Dockerfile . > build.log 2>&1
```

### Gemma 2 9b test

Test the container on a GPU instance (g2) as follows:

```bash
model=google/gemma-2-9b-it
num_shard=4
max_input_length=1562
max_total_tokens=2048

docker run --gpus all -ti --ipc=host -p 8080:8080 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.1.1
```

Send request:

```bash
curl 0.0.0.0:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 128
}' \
    -H 'Content-Type: application/json'
```

### Mistral 7B test

Test the container on a GPU instance (g2) as follows:

```bash
model=HuggingFaceH4/zephyr-7b-beta
num_shard=1
max_input_length=1562
max_total_tokens=2048

docker run --gpus all -ti -p 8080:8080 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.0.2  
```

Send request:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"[INST] What is 10+10? [\/INST]","parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```

### Gemma Test

```bash
model=google/gemma-7b
num_shard=1
max_input_length=512
max_total_tokens=1024
max_batch_prefill_tokens=1512
token=YOUR_TOKEN

docker run --gpus all -ti -p 8080:80 \
  -e MODEL_ID=$model \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -e MAX_BATCH_PREFILL_TOKENS=$max_batch_prefill_tokens \
  -e HUGGING_FACE_HUB_TOKEN=$token \
  us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.4.2
```

Send request:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Deep Learning is a","parameters":{"temperature":1.0, "top_p": 0.95, "max_new_tokens": 256}}' \
    -H 'Content-Type: application/json'
```

For a Vertex AI example checkout [Deploy Gemma on Vertex AI](../../examples/vertex-ai/notebooks/deploy-gemma-on-vertex-ai.ipynb) notebook.
