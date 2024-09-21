# Text Generation Inference (TGI) Containers

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a toolkit developed by Hugging Face for deploying and serving LLMs, with high performance text generation, that comes with a Rust, Python and gRPC server for text generation inference, used in production at Hugging Face to power Hugging Chat, the Inference API and Inference Endpoints.

## Published Containers

To check which of the available Hugging Face DLCs are published, you can either check the [Google Cloud Deep Learning Containers Documentation for TGI](https://cloud.google.com/deep-learning-containers/docs/choosing-container#text-generation-inference), the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or use the `gcloud` command to list the available containers with the tag `huggingface-text-generation-inference` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-text-generation-inference"
```

## Getting Started

Below you will find the instructions on how to run and test the TGI containers available within this repository. Note that before proceeding you need to first ensure that you have Docker installed either on your local or remote instance, if not, please follow the instructions on how to install Docker [here](https://docs.docker.com/get-docker/).

To run the Docker container in GPUs you need to ensure that your hardware is supported (NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher) and also install the NVIDIA Container Toolkit.

To find the supported models and hardware before running the TGI DLC, feel free to check [TGI Documentation](https://huggingface.co/docs/text-generation-inference/supported_models).

### Run

To run this DLC, you need to have GPU accelerators available within the instance that you want to run TGI, not only because those are required, but also to enable the best performance due to the optimized inference CUDA kernels.

Besides that, you also need to define the model to deploy, as well as the generation configuration. For the model selection, you can pick any model from the Hugging Face Hub that contains the tag `text-generation-inference` which means that it's supported by TGI; to explore all the available models within the Hub, please check [here](https://huggingface.co/models?other=text-generation-inference&sort=trending). Then, to select the best configuration for that model you can either keep the default values defined within TGI, or just select the recommended ones based on our instance specification via the Hugging Face Recommender API for TGI as follows:

```bash
curl -G https://huggingface.co/api/integrations/tgi/v1/provider/gcp/recommend \
    -d "model_id=google/gemma-7b-it" \
    -d "gpu_memory=80" \
    -d "num_gpus=2"
```

Which returns the following output containing the optimal configuration for deploying / serving that model via TGI:

```json
{
    "model_id": "google/gemma-7b-it",
    "instance": "g2-standard-4",
    "configuration": {
    "model_id": "google/gemma-7b-it",
    "max_batch_prefill_tokens": 4096,
    "max_input_length": 4000,
    "max_total_tokens": 4096,
    "num_shard": 1,
    "quantize": null,
    "estimated_memory_in_gigabytes": 22.77
}
```

Then you are ready to run the container as follows:

```bash
docker run --gpus all -ti --shm-size 1g -p 8080:8080 \
    -e MODEL_ID=google/gemma-7b-it \
    -e NUM_SHARD=4 \
    -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    -e MAX_INPUT_LENGTH=4000 \
    -e MAX_TOTAL_TOKENS=4096 \
    us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-3.ubuntu2204.py310
```

### Test

Once the Docker container is running, as it has been deployed with `text-generation-launcher`, the API will expose the following endpoints listed within the [TGI OpenAPI Specification](https://huggingface.github.io/text-generation-inference/).

In this case, you can test the container by sending a request to the `/v1/chat/completions` endpoint (that matches OpenAI specification and so on is fully compatible with OpenAI clients) as follows:

```bash
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
```

Which will start streaming the completion tokens for the given messages until the stop sequences are generated.

Alternatively, you can also use the `/generate` endpoint instead, which already expects the inputs to be formatted according to the tokenizer requirements, which is more convenient when working with base models without a pre-defined chat template or whenever you want to use a custom chat template instead, and can be used as follows:

```bash
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

In order to build TGI Docker container, you will need an instance with at least 4 NVIDIA GPUs available with at least 24 GiB of VRAM each, since TGI needs to build and compile the kernels required for the optimized inference. Also note that the build process may take ~30 minutes to complete, depending on the instance's specifications.

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu121.2-3.ubuntu2204.py310 -f containers/tgi/gpu/2.2.0/Dockerfile .
```
