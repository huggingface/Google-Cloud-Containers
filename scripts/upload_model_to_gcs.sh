#!/bin/bash

# This script downloads a model from HuggingFace and uploads it to a bucket in Google Cloud Storage.
# ./upload_model_to_gcs.sh --model-id meta-llama/Meta-Llama-3-8B-Instruct --gcs gs://hf-models-gke-bucket/Meta-Llama-3-8B-Instruct --location us-central1

# Exit immediately if a command exits with a non-zero status
set -e

# Parse command-line arguments for repository and bucket
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-id) REPOSITORY_ID="$2"; shift ;;
        --gcs) GCS_BUCKET="$2"; shift ;;
        # optional
        --location) LOCATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if necessary parameters are provided
if [[ -z "$REPOSITORY_ID" || -z "$GCS_BUCKET" ]]; then
    echo "Usage: $0 --model-id id --gcs gs://bucket-name"
    exit 1
fi

# Ensure the necessary environment variables are set
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create a local directory to store the downloaded models
TMP_DIR="tmp"
LOCAL_DIR="$TMP_DIR/$REPOSITORY_ID"
mkdir -p $LOCAL_DIR

# Download models from HuggingFace, excluding certain file types
echo "Downloading hf.co/$REPOSITORY_ID model files to $LOCAL_DIR..."
huggingface-cli download $REPOSITORY_ID --exclude "*.bin" "*.pth" "*.gguf" ".gitattributes" --local-dir $LOCAL_DIR
if [ $? -ne 0 ]; then
    echo "Download failed with error code $?"
    exit 1
fi
echo "Download successfully completed!"

# Parse the bucket from the provided $GCS_BUCKET path i.e. given gs://bucket-name/dir, extract bucket-name
BUCKET_NAME=$(echo $GCS_BUCKET | cut -d'/' -f3)
# Check if the bucket exists, if not create it
if [ -z "$(gsutil ls | grep gs://$BUCKET_NAME)" ]; then
    echo "Bucket gs://$BUCKET_NAME does not exist, so creating it now..."
    if [ -z "$LOCATION" ]; then
        gcloud storage buckets create gs://$BUCKET_NAME --default-storage-class=STANDARD --uniform-bucket-level-access
    else
        gcloud storage buckets create gs://$BUCKET_NAME --location=$LOCATION --default-storage-class=STANDARD --uniform-bucket-level-access
    fi

    if [ $? -ne 0 ]; then
        echo "Bucket creation failed with error code $?"
        exit 1
    fi
    echo "Bucket $BUCKET_NAME created successfully!"
fi

# Upload the downloaded models to Google Cloud Storage
echo "Uploading model files to $GCS_BUCKET..."
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -e -r $LOCAL_DIR/* $GCS_BUCKET
echo "Upload successfully completed!"

# Remove all files and hidden files in the target directory
find "$LOCAL_DIR" -type f -name '.*' -delete
find "$LOCAL_DIR" -type f -delete
# Remove empty directories within the target directory
find "$LOCAL_DIR" -depth -type d -empty -exec rmdir {} +
# Remove empty directories within the root directory
find "$TMP_DIR" -depth -type d -empty -exec rmdir {} +
echo "Local directory $LOCAL_DIR cleaned up."
