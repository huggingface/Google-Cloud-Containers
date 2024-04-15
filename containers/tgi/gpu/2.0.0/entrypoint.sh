#!/bin/bash

# Check if MODEL_ID starts with "gcs://"
if [[ $AIP_STORAGE_URI == gs://* ]]; then
    echo "AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."

    # Define the target directory
    TARGET_DIR="/tmp/model"
    mkdir -p "$TARGET_DIR"

    # Use gsutil to copy the content from GCS to the target directory
    gsutil -m cp -r "$AIP_STORAGE_URI" "$TARGET_DIR"

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

# launch TGI either from MODEL_ID (HF repo) or local directory
text-generation-launcher --port 8080 --json-output