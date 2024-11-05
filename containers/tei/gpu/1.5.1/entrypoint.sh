#!/bin/bash

# Check if MODEL_ID starts with "gcs://"
if [[ $AIP_STORAGE_URI == gs://* ]]; then
    echo "AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."
    echo "AIP_STORAGE_URI: $AIP_STORAGE_URI"

    # Define the target directory
    TARGET_DIR="/tmp/model"
    mkdir -p "$TARGET_DIR"

    # Use gsutil to copy the content from GCS to the target directory
    echo "Running: gcloud storage storage cp $AIP_STORAGE_URI/* $TARGET_DIR --recursive"
    gcloud storage cp "$AIP_STORAGE_URI/*" "$TARGET_DIR" --recursive

    # Check if gsutil command was successful
    if [ $? -eq 0 ]; then
        echo "Model downloaded successfully to ${TARGET_DIR}."
        # Update MODEL_ID to point to the local directory
        echo "Updating MODEL_ID to point to the local directory."
        export MODEL_ID="$TARGET_DIR"
    else
        echo "Failed to download model from GCS."
        exit 1
    fi
fi

ldconfig 2>/dev/null || echo "unable to refresh ld cache, not a big deal in most cases"

# Below is the original `cuda-all-entrypoint.sh` script.
# Reference: https://github.com/huggingface/text-embeddings-inference/blob/v1.4.0/cuda-all-entrypoint.sh
if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 "$@"
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec text-embeddings-router-80 "$@"
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 "$@"
else
    echo "cuda compute cap ${compute_cap} is not supported"
    exit 1
fi
