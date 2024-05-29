#!/bin/bash
####################################################################
# Description: Pushes local TGI image to GCR for testing vertex AI #
####################################################################

REGION="us-central1"
# DOCKER_ARTIFACT_REPO="base-tgi-image"
DOCKER_ARTIFACT_REPO="base-infernece-image"
PROJECT_ID="gcp-partnership-412108"
# BASE_TGI_IMAGE="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-gpu.2.0.3"
BASE_TGI_IMAGE="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311"
SERVING_CONTAINER_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_ARTIFACT_REPO}/base-inference-image:latest"

# gcloud auth application-default login
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
echo "Pushing ${SERVING_CONTAINER_IMAGE_URI}"
docker push "${SERVING_CONTAINER_IMAGE_URI}"

