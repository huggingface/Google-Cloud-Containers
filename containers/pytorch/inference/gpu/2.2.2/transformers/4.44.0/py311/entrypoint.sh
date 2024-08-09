#!/bin/bash

# Define the default port
PORT=5000

# Check if AIP_MODE is set and adjust the port for Vertex AI
if [[ ! -z "${AIP_MODE}" ]]; then
    PORT=${AIP_HTTP_PORT}
fi

# Check if MODEL_ID starts with "gcs://"
if [[ $AIP_STORAGE_URI == gs://* ]]; then
    echo "AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."
    echo "AIP_STORAGE_URI: $AIP_STORAGE_URI"

    # Define the target directory
    TARGET_DIR="/opt/huggingface/model"
    mkdir -p "$TARGET_DIR"

    # Use gsutil to copy the content from GCS to the target directory
    echo "Running: gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -e -r "$AIP_STORAGE_URI/*" "$TARGET_DIR""
    gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -e -r "$AIP_STORAGE_URI/*" "$TARGET_DIR"

    # Check if gsutil command was successful
    if [ $? -eq 0 ]; then
        echo "Model downloaded successfully to ${TARGET_DIR}."
        # Update MODEL_ID to point to the local directory
        echo "Updating MODEL_ID to point to the local directory."
        export HF_MODEL_DIR="$TARGET_DIR"
        export AIP_STORAGE_URI=""
    else
        echo "Failed to download model from GCS."
        exit 1
    fi
fi

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

# Start the server
uvicorn huggingface_inference_toolkit.webservice_starlette:app --host 0.0.0.0 --port ${PORT}