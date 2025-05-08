# Hugging Face PyTorch Inference Containers

The Hugging Face PyTorch Inference containers are Python containers for serving 🤗`transformers` models with FastAPI on Google Cloud AI Platform. There are two containers at the moment, one for CPU and one for GPU; and one for TPU (which is coming soon).

The PyTorch Inference containers are powered by [huggingface-inference-toolkit](https://github.com/huggingface/huggingface-inference-toolkit) which is a Python package used to serve `transformers`, `sentence-transformers`, and `diffusers` models; for Google Cloud, defines a Custom Prediction Routine (CPR) for serving custom models in Vertex AI and Google Kubernetes Engine (GKE).

The [huggingface-inference-toolkit](https://github.com/huggingface/huggingface-inference-toolkit) provides default pre-processing, predict and post-processing methods for `transformers` and `sentence-transformers`, while also enabling the definition of a custom handler for customizing pre-processing and post-processing steps.

> [!NOTE]
> These containers are named PyTorch containers since PyTorch is the backend framework used for training the models; but it comes with all the required Hugging Face libraries installed.

## Published Containers

To check which of the available Hugging Face DLCs are published, you can either check the [Google Cloud Deep Learning Containers Documentation for PyTorch Inference](https://cloud.google.com/deep-learning-containers/docs/choosing-container#pytorch-inference), the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or use the `gcloud` command to list the available containers with the tag `huggingface-pytorch-inference` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-pytorch-inference"
```

## Getting Started

Below you will find the instructions on how to run and test the PyTorch Inference containers available within this repository. Note that before proceeding you need to first ensure that you have Docker installed either on your local or remote instance, if not, please follow the instructions on how to install Docker [here](https://docs.docker.com/get-docker/).

Additionally, if you're willing to run the Docker container in GPUs you will need to install the NVIDIA Container Toolkit.

## Run

Before running this container, you will need to select any supported model from the Hugging Face Hub offering for [`transformers`](https://huggingface.co/models?library=transformers&sort=trending), [`diffusers`](https://huggingface.co/models?library=diffusers&sort=trending), and [`sentence-transformers`](https://huggingface.co/models?library=sentence-transformers&sort=trending), as well as the task that the model runs.

The Hugging Face PyTorch DLCs for Inference come with a pre-defined entrypoint, so to run those you only need to define the environment variable values of the model and task that you want to deploy, being the `HF_MODEL_ID` and `HF_TASK` respectively. Besides those, you can also define a wide range of environment variable values supported within the [`huggingface-inference-toolkit`](https://github.com/huggingface/huggingface-inference-toolkit) as detailed [here](https://github.com/huggingface/huggingface-inference-toolkit?tab=readme-ov-file#%EF%B8%8F-environment-variables).

> [!NOTE]
> As [huggingface-inference-toolkit](https://github.com/huggingface/huggingface-inference-toolkit) is built to be fully compatible with Google Vertex AI, then you can also set the environment variables defined by Vertex AI such as `AIP_MODE=PREDICTION`, `AIP_HTTP_PORT=8080`, `AIP_PREDICT_ROUTE=/predict`, `AIP_HEALTH_ROUTE=/health`, and some more. To read about all the exposed environment variables in Vertex AI please check [Vertex AI Documentation - Custom container requirements for prediction](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables).

### Supported Tasks

You can find a list of the supported tasks with a brief introduction, links to the documentation and an example on how to use those within the Hugging Face PyTorch DLC for Inference [here](./TASKS.md).

### Supported Hardware

The Hugging Face PyTorch DLCs for Inference are available for both CPU and GPU, and you can select the container based on the hardware you have available.

- **CPU**

  ```bash
  docker run -ti -p 5000:5000 \
      -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
      -e HF_TASK=text-classification \
      --platform linux/amd64 \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-3.transformers.4-46.ubuntu2204.py311
  ```

- **GPU**: Note that here you need to have an instance with at least one NVIDIA GPU and to set the `--gpus all` flag within the `docker run` command, as well as using the GPU-compatible container.

  ```bash
  docker run -ti --gpus all -p 5000:5000 \
      -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
      -e HF_TASK=text-classification \
      --platform linux/amd64 \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311
  ```

## Test

Once the Docker container is running, you can start sending requests to the `/predict` endpoint which is the default endpoint exposed by the Hugging Face PyTorch DLCs for Inference (unless overridden with `AIP_PREDICT_ROUTE` on run time).

```bash
curl http://0.0.0.0:5000/predict \
    -X POST \
    -H 'Content-Type: application/json; charset=UTF-8' \
    -d '{
        "inputs": ["I love this product", "I hate this product"],
        "parameters": { "top_k": 2 }
    }'
```

> [!NOTE]
> You can see which are the expected input and output payloads for each task, as that's conditioned by the `HF_TASK` environment variable value set during the `docker run` command, [here](./TASKS.md).

## Optional

### Build

> [!WARNING]
> Building the containers is not recommended since those are already built by Hugging Face and Google Cloud teams and provided openly, so the recommended approach is to use the pre-built containers available in [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) instead.

The PyTorch Training containers come with two different containers depending on the accelerator used for training, being either CPU or GPU, but those can be built within the same instance, that does not need to have a GPU available.

- **CPU**

  ```bash
  docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-3.transformers.4-46.ubuntu2204.py311 -f containers/pytorch/inference/cpu/2.3.1/transformers/4.46.1/py311/Dockerfile --platform linux/amd64 .
  ```

- **GPU**

  ```bash
  docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311 -f containers/pytorch/inference/gpu/2.3.1/transformers/4.46.1/py311/Dockerfile --platform linux/amd64 .
  ```
