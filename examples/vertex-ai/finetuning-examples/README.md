# Instructions for building the docker image and testing it by running the notebook
This file contains step by step instructions on how to build the docker image so that one can test the model and the notebook. 


### Pre-requisites:
1. Access to [gg-hf](https://huggingface.co/gg-hf) on Hugging Face Hub in order to download the model and the tokenizer.
2. Access to [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers) GitHub repository in order to access the docker file.
3. Access to [new-model-addition-golden-gate](https://github.com/huggingface/new-model-addition-golden-gate) GitHub repository in order to use transformer library with the gg-hf model integrated into it.


We use the [golden-gate-finetuning-clm-lora-sft.ipynb](https://github.com/huggingface/Google-Cloud-Containers/tree/add-example-notebook/examples/vertex-ai/finetuning-examples/golden-gate-finetuning-clm-lora-sft.ipynb) Notebook to test the model.


### Steps to build the docker image:
 
1. Clone the [Google-Cloud-Containers](https://github.com/huggingface/Google-Cloud-Containers)

2. Build the docker image using the following command:
```bash
cd Google-Cloud-Containers
docker build -t pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310 -f containers/pytorch/training/gpu/2.1/transformers/4.37.2/py310/Dockerfile .
```

3. You can choose one of the following method to install the transformers library with the gg-hf model integrated into it:

a. Using the existing image built in the previous step and then manually installing after running the container. This method is illustrated in Method 1: Pushing the image to the Google Cloud Artifact registry and running the notebook on Vertex AI and Method 2: Running the notebook locally.

b. Modifying the Dockerfile to use the latest version of the transformers library and then building the image again. For this method, you can follow the instructions in the section Modify Dockerfile to use the latest version of the transformers library. Add the following lines to the Dockerfile
  ```Dockerfile
  ARG GITHUB_TOKEN="" # Github token to access the private repository, define it while building the image. Read more about it here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens

  RUN pip install git+https://${GITHUB_TOKEN}@github.com/huggingface/new-model-addition-golden-gate.git # After the line that installs the transformers library
  ```

  Build the docker image using the following command:
  ```bash

  docker build --build-arg="GITHUB_TOKEN=xxxxx" -t pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310 -f containers/pytorch/training/gpu/2.1/transformers/4.37.2/py310/Dockerfile .
  ```

If you choose the second method, you can then test the model directly by running the notebook on the docker container as listed below in the section Steps to test the docker image.

### Steps to test the docker image:
You can choose one of the following methods to test the docker image:

#### Method 1: Pushing the image to the Google Cloud Artifact registry and running the notebook on Vertex AI
You can push the image to the Google Cloud Artifact registry using the following script:

```bash
#!/bin/bash
####################################################################
# Description: Pushes local Deep Learning image to GAR for testing vertex AI 
####################################################################

REGION=""
DOCKER_ARTIFACT_REPO=""
PROJECT_ID=""
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

Only needed when you haven't modified the Dockerfile to use the latest version of the transformers library. If you have modified the Dockerfile, you can skip this step.
You can then spin up a Vertex AI Workbench and make sure to choose the image from the Artifact registry. In that notebook, you can install the transformers library with the gg-hf model integrated into it. You can do this by running the following command inside the notebook:
```bash
git clone https://github.com/huggingface/new-model-addition-golden-gate
cd new-model-addition-golden-gate
git checkout add-golden-gate
pip install -e .
cd ..
```
Copy the notebook file to the Vertex AI Workbench and test the model. 

#### Method 2: Running the notebook locally
Make sure you have the notebook file on your local machine. As we are mounting the current directory to the docker container.

```bash
docker run -it --gpus all -p 8080:8080 -v $(pwd):/workspace pytorch-training-gpu.2.1.transformers.4.38.0.dev0.py310
```
Only needed when you haven't modified the Dockerfile to use the latest version of the transformers library. If you have modified the Dockerfile, you can skip this step.
Install the transformers library with the gg-hf model integrated into it. You can do this by running the following command inside the docker container:
```bash
git clone https://github.com/huggingface/new-model-addition-golden-gate
cd new-model-addition-golden-gate
git checkout add-golden-gate
pip install -e .
cd ..
```

Inside the docker container, you can run the following command to start the jupyter notebook:
```bash
jupyter notebook --ip 0.0.0.0 --port 8080 --no-browser --allow-root
```
Access the notebook through your desktops browser on the link generated by the above command, which will look something like this:
http://hostname:8888/?token=4843b2c524efb137a08421202ddc7c40acc1bb9ee151dbee

```
Now, you can run the notebook and test the model with the latest version of the transformers library.
