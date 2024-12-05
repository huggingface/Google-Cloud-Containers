# Text Generation Inference (TGI) Containers

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a toolkit developed by Hugging Face for deploying and serving LLMs, with high performance text generation, that comes with a Rust, Python and gRPC server for text generation inference, used in production at Hugging Face to power Hugging Chat, the Inference API and Inference Endpoints.

## Published Containers

To check which of the available Hugging Face DLCs are published, you can either check the [Google Cloud Deep Learning Containers Documentation for TGI](https://cloud.google.com/deep-learning-containers/docs/choosing-container#text-generation-inference), the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or use the `gcloud` command to list the available containers with the tag `huggingface-text-generation-inference` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-text-generation-inference"
```

## Getting Started

Below you will find the instructions on how to run and test the TGI containers available within this repository. Note that before proceeding you need to first ensure that you have Docker installed either on your local or remote instance, if not, please follow the instructions on how to install Docker [here](https://docs.docker.com/get-docker/).

### Run

The TGI containers support two different accelerator types: GPU and TPU. Depending on your infrastructure, you'll use different approaches to run the containers.

- **GPU**: To run this DLC, you need to have GPU accelerators available within the instance that you want to run TGI, not only because those are required, but also to enable the best performance due to the optimized inference CUDA kernels.

  To find the supported models and hardware before running the TGI DLC, feel free to check [TGI Documentation](https://huggingface.co/docs/text-generation-inference/supported_models).

  First, you can use the Hugging Face Recommender API to get the optimal configuration:

  ```bash
  curl -G https://huggingface.co/api/integrations/tgi/v1/provider/gcp/recommend \
      -d "model_id=google/gemma-7b-it" \
      -d "gpu_memory=24" \
      -d "num_gpus=1"
  ```

  Then run the container:

  ```bash
  docker run --gpus all -ti --shm-size 1g -p 8080:8080 \
      -e MODEL_ID=google/gemma-7b-it \
      -e NUM_SHARD=1 \
      -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
      -e MAX_INPUT_LENGTH=4000 \
      -e MAX_TOTAL_TOKENS=4096 \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-3.ubuntu2204.py311
  ```

- **TPU**: This example showcases how to deploy a TGI server on a TPU instance using the TGI container. Note that TPU support for TGI is currently experimental and may have limitations compared to GPU deployments.

  ```bash
  docker run --rm --net host --privileged \
      -p 8080:8080 \
      -e MODEL_ID=google/gemma-2b \
      -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
      -e MAX_INPUT_LENGTH=2048 \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-tpu.2.4.0.py310
  ```

  > [!NOTE]
  > TPU support for Text Generation Inference is still evolving. Check the [Hugging Face TPU documentation](https://huggingface.co/docs/optimum-tpu/) for the most up-to-date information on TPU model serving.

### Test

Once the Docker container is running, you can test it by sending requests to the available endpoints.

For the GPU/TPU container running on localhost, you can use the following curl commands:

```bash
# Chat Completions Endpoint
curl 0.0.0.0:8080/v1/chat/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "tgi",
        "messages": [
            {
                "role": "user",
                "content": "What is Deep Learning?"
            }
        ],
        "stream": true,
        "max_tokens": 128
    }'

# Generate Endpoint
curl 0.0.0.0:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "inputs": "What is Deep Learning?",
        "parameters": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_new_tokens": 256
        }
    }'
```

## Optional

### Build

> [!WARNING]
> Building the containers is not recommended since those are already built by Hugging Face and Google Cloud teams and provided openly, so the recommended approach is to use the pre-built containers available in [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) instead.

The TGI containers come with two different variants depending on the accelerator used:

- **GPU**: To build the TGI container for GPU, you will need an instance with at least 4 NVIDIA GPUs available with at least 24 GiB of VRAM each, since TGI needs to build and compile the kernels required for the optimized inference. The build process may take ~30 minutes to complete.

  ```bash
  docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-3.ubuntu2204.py311 -f containers/tgi/gpu/2.3.1/Dockerfile .
  ```

- **TPU**: To build the TGI container for Google Cloud TPUs, an instance with at least one TPU available is required. The build process may have specific requirements for TPU-compatible libraries.

  ```bash
  docker build --ulimit nofile=100000:100000 -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-tpu.0.2.1.py310 -f containers/tgi/tpu/0.2.1/Dockerfile .
  ```

  to run the TGI server

  ```bash
    docker run --rm -p 8080:80 \
        --shm-size 16G --ipc host --privileged  \
        -e HF_TOKEN=${HF_TOKEN} \
        -e SKIP_WARMUP=1 \
        us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-tpu.0.2.1.py310 \
        --model-id google/gemma-2b-it
        --max-input-length 512 \
        --max-total-tokens 1024 \
        --max-batch-prefill-tokens 512 \
        --max-batch-total-tokens 1024
  ```
google/gemma-2-2b-it
  openai-community/gpt2

  It can then be queried like this 

  ```bash
  curl 0.0.0.0:80/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
        "inputs": "What is Deep Learning?",
        "parameters": {
            "max_new_tokens": 20
        }
    }'
  ```