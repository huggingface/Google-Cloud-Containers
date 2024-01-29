#!/bin/bash
####################################################################
# Description: Pushes local TGI image to GCR for testing vertex AI #
####################################################################

REGION="us-central1"
DOCKER_ARTIFACT_REPO="custom-tgi-example"
PROJECT_ID="huggingface-ml"
BASE_TGI_IMAGE="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.1.3.4"
SERVING_CONTAINER_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_ARTIFACT_REPO}/base-tgi-image:latest"

gcloud auth login
gcloud auth application-default login
gcloud config set project "${PROJECT_ID}" --quiet
gcloud config set ai/region "${REGION}" --quiet

gcloud services enable artifactregistry.googleapis.com

# create a new Docker repository with your region with the description
gcloud artifacts repositories create "${DOCKER_ARTIFACT_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Custom TGI Example"

# verify that your repository was created.
gcloud artifacts repositories list \
  --location="${REGION}" \
  --filter="name~${DOCKER_ARTIFACT_REPO}"

# configure docker to use your repository    
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# pull, tag and push
docker tag "${BASE_TGI_IMAGE}" "${SERVING_CONTAINER_IMAGE_URI}"
docker push "${SERVING_CONTAINER_IMAGE_URI}"
