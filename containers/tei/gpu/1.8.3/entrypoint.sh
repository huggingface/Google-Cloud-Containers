#!/bin/bash

ldconfig 2>/dev/null || echo "WARN: Unable to refresh ld cache, not a big deal in most cases"

# NOTE: We first check that whether the `nvidia-smi` command is found, as otherwise
# it makes no sense to download the artifacts from Google Cloud Storage. In this
# case, we could eventually suggest the user to either use an NVIDIA GPU instance
# on Google Cloud, or rather to use the Text Embeddings Inference (TEI) CPU DLC instead.
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi command not found."
    exit 1
fi

if [[ $AIP_STORAGE_URI == gs://* ]]; then
    echo "INFO: Provided AIP_STORAGE_URI=$AIP_STORAGE_URI, which is a Google Cloud Storage path given that it starts with gs://"

    TARGET_DIR="/tmp/model"
    mkdir -p "$TARGET_DIR"

    echo "INFO: gcloud storage storage cp $AIP_STORAGE_URI/* $TARGET_DIR --recursive"
    gcloud storage cp "$AIP_STORAGE_URI/*" "$TARGET_DIR" --recursive

    if [ $? -eq 0 ]; then
        echo "INFO: Model downloaded successfully to ${TARGET_DIR}."
        # NOTE: Update MODEL_ID to point to the local directory once downloaded
        echo "INFO: Updating MODEL_ID to point to the local directory."
        export MODEL_ID="$TARGET_DIR"
    else
        echo "ERROR: Failed to download model from GCS."
        exit 1
    fi
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')
if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 "$@"
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec text-embeddings-router-80 "$@"
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 "$@"
else
    echo "ERROR: CUDA compute capability ${compute_cap} is not supported"
    exit 1
fi
