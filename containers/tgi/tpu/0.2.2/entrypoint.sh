#!/bin/bash

# This is required by GKE, see
# https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#privileged-mode
ulimit -l 68719476736

# Check if MODEL_ID starts with "gs://"
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

if [[ -z "${MAX_BATCH_SIZE}" ]]; then
  # Default to a batch size of 4 if no value is provided
  export MAX_BATCH_SIZE="4"
fi

if [[ -n "${QUANTIZATION}" ]]; then
  # If quantization is set, we use jetstream_int8 (this is the only option supported by optimum-tpu at the moment)
  QUANTIZATION="jetstream_int8"
  export QUANTIZATION="${QUANTIZATION}"
fi

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

exec text-generation-launcher $@
