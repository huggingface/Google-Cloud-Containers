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

Before running this container, you will need to select any supported model from the [Hugging Face Hub offering for `transformers`](https://huggingface.co/models?library=transformers&sort=trending), as well as the task that the model runs as e.g. text-classification.

* **CPU**

    ```bash
    docker run -ti -p 8080:8080 \
        -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
        -e HF_TASK=text-classification \
        us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311
    ```

* **GPU**: Note that here you need to have an instance with at least one NVIDIA GPU and to set the `--gpus all` flag within the `docker run` command, as well as using the GPU-compatible container.

    ```bash
    docker run -ti --gpus all -p 8080:8080 \
        -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
        -e HF_TASK=text-classification \
        us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311
    ```

> [!NOTE]
> As [huggingface-inference-toolkit](https://github.com/huggingface/huggingface-inference-toolkit) is built to be fully compatible with Google Vertex AI, then you can also set the environment variables defined by Vertex AI such as `AIP_MODE=PREDICTION`, `AIP_HTTP_PORT`, `AIP_PREDICT_ROUTE`, `AIP_HEALTH_ROUTE`, and some more. To read about all the exposed environment variables in Vertex AI please check [Vertex AI Documentation - Custom container requirements for prediction](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables).

## Test

Once the Docker container is running, you can start sending requests to the `/predict` endpoint which is the default endpoint exposed by the PyTorch Inference containers (unless overridden with `AIP_PREDICT_ROUTE` on build time).

```bash
curl 0.0.0.0:8080/predict \
    -X POST \
    -H 'Content-Type: application/json; charset=UTF-8' \
    -d '{
        "instances": ["I love this product", "I hate this product"],
        "parameters": { "top_k": 2 }
    }'
```

> [!NOTE]
> The [huggingface-inference-toolkit`(https://github.com/huggingface/huggingface-inference-toolkit) is powered by the `pipeline` method within `transformers`, that means that the payload will be different based on the model that you're deploying. So on, before sending requests to the deployed model, you will need to first check which is the task that the `pipeline` method and the model support and are running. To read more about the `pipeline` and the supported tasks please check [Transformers Documentation - Pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines).

## Optional

### Build

> [!WARNING]
> Building the containers is not recommended since those are already built by Hugging Face and Google Cloud teams and provided openly, so the recommended approach is to use the pre-built containers available in [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) instead.

The PyTorch Training containers come with two different containers depending on the accelerator used for training, being either CPU or GPU, but those can be built within the same instance, that does not need to have a GPU available.

* **CPU**

    ```bash
    docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
    ```

* **GPU**

    ```bash
    docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
    ```
