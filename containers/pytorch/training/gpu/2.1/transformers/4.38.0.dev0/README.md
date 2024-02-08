# Fine-tune Gemma-2B on Vertex AI WorkBench

This file contains step by step instructions on how to build a docker image and then run it to test the Gemma-2B model using the 
[gemma-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb) Notebook on a Vertex AI WorkBench instance and on your local machine.

## Pre-requisites:
1. Access to [gg-hf](https://huggingface.co/gg-hf) on Hugging Face Hub in order to download the model and the tokenizer.
2. Access to [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers) GitHub repository in order to access the docker file.
3. Access to [new-model-addition-gemma](https://github.com/huggingface/new-model-addition-gemma) GitHub repository in order to use transformer library with the gg-hf model integrated into it.


We use the [gemma-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb) Notebook to test the model.

## Build the docker image

Use the following command to build the docker image. Make sure to replace the value of `GITHUB_TOKEN` with your own token.

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
cd Google-Cloud-Containers
docker build --build-arg="GITHUB_TOKEN=xxxxx" -t pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310 -f containers/pytorch/training/gpu/2.1/transformers/4.38.0.dev0/py310/Dockerfile .
```

For setting the value of `GITHUB_TOKEN` please follow the detailed instructions mentioned in the following links: 
- [Creating a fine-grained personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)

- [Creating a personal access token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)


## Using Vertex AI WorkBench Instance to fine-tune the Gemma-2B model

It consists of the following steps:
1. Push the docker image to the Google Cloud Artifact registry.
2. Create a Vertex AI WorkBench instance.


In order to use Google Cloud, you need to have a Google Cloud project and a Google Cloud account. Make sure to install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk) and authenticate your account using the following command:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project "gcp-project-id"
```

### Push the image to the Google Cloud Artifact registry

Now, you can push the image to the Google Cloud Artifact registry using the following script, make sure to replace the variables with your own values:

```bash
#!/bin/bash
####################################################################
# Description: Pushes local Deep Learning image to GAR for testing vertex AI 
####################################################################

REGION="us-central1"
DOCKER_ARTIFACT_REPO="deep-learning-images"
PROJECT_ID="gcp-project-id"
BASE_IMAGE="pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310"
FRAMEWORK="pytorch"
TYPE="training"
ACCELERATOR="gpu"
FRAMEWORK_VERSION="2.1"
TRANSFORMERS_VERISON="4.38.0.dev0"
PYTHON_VERSION="py310"

SERVING_CONTAINER_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_ARTIFACT_REPO}/huggingface-${FRAMEWORK}-${TYPE}-${ACCELERATOR}.${FRAMEWORK_VERSION}.transformers.${TRANSFORMERS_VERISON}.${PYTHON_VERSION}:latest"

gcloud auth login
gcloud auth application-default login
gcloud config set project "${PROJECT_ID}" --quiet
gcloud config set ai/region "${REGION}" --quiet

gcloud services enable artifactregistry.googleapis.com

create a new Docker repository with your region with the description
gcloud artifacts repositories create "${DOCKER_ARTIFACT_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Deep Learning Images"

# verify that your repository was created.
gcloud artifacts repositories list \
  --location="${REGION}" \
  --filter="name~${DOCKER_ARTIFACT_REPO}"

# configure docker to use your repository    
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# # pull, tag and push
docker tag "${BASE_IMAGE}" "${SERVING_CONTAINER_IMAGE_URI}"
docker push "${SERVING_CONTAINER_IMAGE_URI}"

```

Once the image is pushed to the Google Cloud Artifact registry, you can access it from the Google Cloud console. To use the image, you can create a Vertex AI WorkBench instance and then use the [gemma-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb) Notebook to fine-tune the Gemma model.


### Create a Vertex AI WorkBench instance

There are three different ways to create a Vertex AI Managed Notebook instance:
 - Using the Google Cloud Console
 - Using the Google Cloud CLI
 - Using Terraform

To read in detail, follow [Create a Vertex AI Workbench instance](https://cloud.google.com/vertex-ai/docs/workbench/instances/create).

We will use the Google Cloud CLI to create a Vertex AI WorkBench instance from a Google Cloud Artifact registry image. Use the following command:

```bash
gcloud notebooks instances create example-instance-1 \
    --container-repository=us-central1-docker.pkg.dev/gcp-project-id/deep-learning-images/huggingface-pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310 \
    --container-tag=latest \
    --machine-type=n1-standard-4 \
    --location=us-central1-c \
    --accelerator-core-count=1 \
    --accelerator-type=NVIDIA_TESLA_T4 \
    --data-disk-size=512 \
    --install-gpu-driver

```
You must replace the values of the flags with your own values. More about flags [here](https://cloud.google.com/sdk/gcloud/reference/notebooks/instances/create).

You can access the instance through the link generated by the `gcloud notebooks instances create` command, which will look something like this: `https://notebooks.googleapis.com/v2/projects/gcp-project-id/locations/us-central1-c/operations/operation-xxxx`. It will take you to the Google Cloud Vertex AI Workbench console and you can access the instance from there.

### Fine-tune the Gemma model
Now, we are ready to fine-tune the Gemma model. Inside the instance, clone the [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers) GitHub repository using

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
```

Then, you can access the [gemma-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb) at `Google-Cloud-Containers/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb` Make sure to select the right kernel to run the notebook.

## Using your Local Machine to fine-tune the Gemma model

Make sure you have the [gemma-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/gemma-finetuning-clm-lora-sft.ipynb) Notebook on your local machine. As we are mounting the current directory to the docker container.

```bash
docker run -it --gpus all -p 8080:8080 -v $(pwd):/workspace pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310
```

Inside the docker container, you can run the following command to start the jupyter notebook:
```bash
jupyter notebook --ip 0.0.0.0 --port 8080 --no-browser --allow-root
```

Note: If you are using a Google Compute Engine instance as your local machine, then in order to access the instance you need to create an SSH tunnel to the instance. You can do this by running the following command on your local machine:
```bash
ssh -N -f -L localhost:8080:localhost:8080 -i PATH_TO_PRIVATE_KEY USERNAME@EXTERNAL_IP
```

Access the notebook through the link generated by the `jupyter notebook` command, which will look something like this:
http://localhost:8080/?token=4843b2c524efb137a08421202ddc7c40acc1bb9ee151dbee