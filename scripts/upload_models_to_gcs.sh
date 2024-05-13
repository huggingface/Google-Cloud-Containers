#!/bin/bash

# This script downloads models from HuggingFace and uploads them to Google Cloud Storage.
# ./upload_models_to_gcs.sh --model-id meta-llama/Meta-Llama-3-8B-Instruct --gcs gs://hf-gcp-models-deployments-test-3451451

# Parse command-line arguments for repository and bucket
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-id) REPOSITORY_ID="$2"; shift ;;
        --gcs) GCS_BUCKET="$2"; shift ;;
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
LOCAL_DIR="tmp/$REPOSITORY_ID"  # The directory where models will be downloaded

# Download models from HuggingFace, excluding certain file types
mkdir -p $LOCAL_DIR
huggingface-cli download $REPOSITORY_ID --exclude "*.bin" "*.pth" "*.gguf" --local-dir $LOCAL_DIR

# Upload the downloaded models to Google Cloud Storage
gsutil -m cp -r $LOCAL_DIR $GCS_BUCKET

# Clean up local directory after upload
rm -rf $LOCAL_DIR

echo "Model download and upload completed."