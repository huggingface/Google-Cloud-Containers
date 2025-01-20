#!/bin/bash

set -eo pipefail

# Define the default port
readonly DEFAULT_PORT=5000

# Check if `AIP_MODE` is set and adjust the port for Vertex AI
if [[ ! -z "${AIP_MODE}" ]]; then
    PORT="${AIP_HTTP_PORT:-$DEFAULT_PORT}"
else
    PORT="$DEFAULT_PORT"
fi

# Just one of `HF_MODEL_ID`, `HF_MODEL_DIR` and `AIP_STORAGE_URI` can be provided
if [[ $(( ${#HF_MODEL_ID:+1} + ${#HF_MODEL_DIR:+1} + ${#AIP_STORAGE_URI:+1} )) -gt 1 ]]; then
    echo "ERROR: Only one of HF_MODEL_ID, HF_MODEL_DIR, or AIP_STORAGE_URI should be provided." >&2
    exit 1
elif [[ $(( ${#HF_MODEL_ID:+1} + ${#HF_MODEL_DIR:+1} + ${#AIP_STORAGE_URI:+1} )) -eq 0 ]]; then
    echo "ERROR: At least one of HF_MODEL_ID, HF_MODEL_DIR, or AIP_STORAGE_URI must be provided." >&2
    exit 1
fi

# If `HF_MODEL_ID` is a path instead of a Hub ID, then clear its value and assign it
# to the `HF_MODEL_DIR` instead, including a user warning
if [[ -d "${HF_MODEL_ID:-}" ]]; then
    echo "WARNING: HF_MODEL_ID is a path, please use HF_MODEL_DIR for paths instead."
    HF_MODEL_DIR="${HF_MODEL_ID}"
    HF_MODEL_ID=""
fi

# Check if `MODEL_ID` starts with "gcs://"
if [[ "${AIP_STORAGE_URI:-}" == gs://* ]]; then
    echo "INFO: AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."
    echo "INFO: AIP_STORAGE_URI: $AIP_STORAGE_URI"

    # Define the target directory
    TARGET_DIR="/opt/huggingface/model"
    mkdir -p "$TARGET_DIR"

    # Check if `gsutil` is available
    if ! command -v gsutil &> /dev/null; then
        echo "ERROR: gsutil command not found. Please install Google Cloud SDK." >&2
        exit 1
    fi

    # Use `gsutil` to copy the content from GCS to the target directory
    echo "INFO: Running: gsutil -m cp -e -r \"$AIP_STORAGE_URI/*\" \"$TARGET_DIR\""
    if ! gsutil -m cp -e -r "$AIP_STORAGE_URI/*" "$TARGET_DIR"; then
        echo "ERROR: Failed to download model from GCS." >&2
        exit 1
    fi

    echo "INFO: Model downloaded successfully to ${TARGET_DIR}."
    echo "INFO: Updating HF_MODEL_DIR to point to the local directory."
    HF_MODEL_DIR="$TARGET_DIR"
    AIP_STORAGE_URI=""
fi

# If `HF_MODEL_DIR` is set and valid, then install the `requirements.txt` file (if available)
if [[ -n "${HF_MODEL_DIR:-}" ]]; then
    # Check if `requirements.txt` exists and if so install dependencies
    if [[ -f "${HF_MODEL_DIR}/requirements.txt" ]]; then
        echo "INFO: Installing custom dependencies from ${HF_MODEL_DIR}/requirements.txt"
        pip install -r "${HF_MODEL_DIR}/requirements.txt" --no-cache-dir

        # Check if `handler.py` is missing when `requirements.txt` is present
        if [[ ! -f "${HF_MODEL_DIR}/handler.py" ]]; then
            echo "WARNING: requirements.txt is present, but handler.py is missing in ${HF_MODEL_DIR}."
            echo "WARNING: If you intend to run custom code, make sure to include handler.py."
        fi
    fi
else
    echo "ERROR: Provided HF_MODEL_DIR is not a valid directory" >&2
    exit 1
fi

# Start the server
exec uvicorn huggingface_inference_toolkit.webservice_starlette:app --host 0.0.0.0 --port "${PORT}"
