#!/bin/bash

ldconfig 2>/dev/null || echo "WARN: Unable to refresh ld cache, not a big deal in most cases"

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

exec text-embeddings-router $@
