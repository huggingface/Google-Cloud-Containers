# Text Generation Inference (TGI) Containers

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a toolkit developed by Hugging Face for deploying and serving LLMs, with high performance text generation, that comes with a Rust, Python and gRPC server for text generation inference, used in production at Hugging Face to power Hugging Chat, the Inference API and Inference Endpoints.

## Published Containers

In order to check which of the available containers are published in Google Cloud's Artifact Registry publicly, you can run the following `gcloud` command to list the available containers with the tag containing `huggingface-text-generation-inference` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-text-generation-inference"
```

## Getting Started

Below you will find the instructions on how to build, run and test the TGI containers available within this repository. Note that before proceeding you need to first ensure that you have Docker installed either on your local or remote instance, if not, please follow the instructions on how to install Docker [here](https://docs.docker.com/get-docker/).

Additionally, if we're willing to build and run the Docker container in GPUs we need to ensure that your hardware is supported (NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher) and also install the NVIDIA Container Toolkit.

To find the supported models and hardware before building and running the TGI image, feel free to check [TGI's documentation](https://huggingface.co/docs/text-generation-inference/supported_models).

### Build

In order to build TGI's Docker container, we will need an instance with at least 4 NVIDIA GPUs available with at least 24 GiB of VRAM each, since TGI needs to build and compile the kernels required for the optimized inference. Also note that the build process may take ~30 minutes to complete, depending on the instance's specifications.

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.2.0 -f containers/tgi/gpu/2.2.0/Dockerfile .
```

Alternatively, you can skip the build process and use the pre-built container available in Google Cloud's Artifact Registry.

### Run

Once the Docker container is built, we can proceed to run it. In this case it's important to have GPUs available within the instance that we want to run TGI, since the GPU accelerators are recommended to enable the best performance due to the optimized inference CUDA kernels.

Besides that, we also need to define the model that we want to deploy, as well as the configuration that we want that model to use. For the model selection, we can pick any model from the Hugging Face Hub that contains the tag `text-generation-inference` which means that it's supported by TGI; to explore all the available models within the Hub, please check [here](https://huggingface.co/models?other=text-generation-inference&sort=trending). Then, to select the best configuration for that model we can either keep the default values defined within TGI, or just select the recommended ones based on our instance specification, and for that we will be using the Hugging Face Recommender API for TGI as follows:

```bash
curl https://huggingface.co/api/integrations/tgi/v1/provider/hf/recommend
    -X GET
    -d "model_id=google/gemma-7b-it"
    -d "gpu_memory=80"
    -d "num_gpus=2"
```

Which returns the following output containing the optimal configuration for deploying / serving that model via TGI:

```json
{
    "model_id": "google/gemma-7b-it",
    "instance": "aws-nvidia-a10g-x1",
    "configuration": {
        "model_id": "google/gemma-7b-it",
        "max_batch_prefill_tokens": 4096,
        "max_input_length": 4000,
        "max_total_tokens": 4096,
        "num_shard": 1,
        "quantize": null,
        "estimated_memory_in_gigabytes": 22.77
    }
}
```

Then we are ready to run the container as follows:

```bash
docker run --gpus all -ti -p 8080:8080 \
    -e MODEL_ID=google/gemma-7b-it \
    -e NUM_SHARD=4 \
    -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    -e MAX_INPUT_LENGTH=4000 \
    -e MAX_TOTAL_TOKENS=4096 \
    us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.1.1
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

Alternatively, we can also use the `/generate` endpoint instead, which already expects the inputs to be formatted according to the tokenizer's requirements, which is more convenient when working with base models without a pre-defined chat template or whenever we want to use a custom chat template instead, and can be used as follows:

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
