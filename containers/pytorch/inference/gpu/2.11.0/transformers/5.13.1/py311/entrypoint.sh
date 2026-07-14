#!/bin/bash

set -eo pipefail

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi command not found, please use the CPU DLC instead."
    exit 1
fi

export CLOUD="google"

# Default `PORT` is 5000, unless other port is provided via `AIP_HTTP_PORT` only
# when `AIP_MODE` is set (usually set to `PREDICT`)
readonly DEFAULT_PORT=5000
if [[ ! -z "${AIP_MODE}" ]]; then
    export PORT="${AIP_HTTP_PORT:-$DEFAULT_PORT}"
else
    export PORT="$DEFAULT_PORT"
fi

if [[ "${AIP_STORAGE_URI:-}" == gs://* ]]; then
    echo "INFO: AIP_STORAGE_URI set and starts with 'gs://', proceeding to download from GCS."
    echo "INFO: AIP_STORAGE_URI: $AIP_STORAGE_URI"

    TARGET_DIR="/opt/huggingface/model"
    mkdir -p "$TARGET_DIR"

    if ! command -v gsutil &>/dev/null; then
        echo "ERROR: gsutil command not found. Please install Google Cloud SDK." >&2
        exit 1
    fi

    echo "INFO: Running: gsutil -m cp -e -r \"$AIP_STORAGE_URI/*\" \"$TARGET_DIR\""
    if ! gsutil -m cp -e -r "$AIP_STORAGE_URI/*" "$TARGET_DIR"; then
        echo "ERROR: Failed to download model from GCS." >&2
        exit 1
    fi

    echo "INFO: Model downloaded successfully to ${TARGET_DIR}."
    echo "INFO: Updating HF_MODEL_DIR to point to the local directory."

    export HF_MODEL_DIR="$TARGET_DIR"
fi

# If `HF_MODEL_ID` is a path instead of a Hub ID, then clear its value and assign it
# to the `HF_MODEL_DIR` instead, including a user warning
if [[ -d "${HF_MODEL_ID:-}" ]]; then
    echo "WARNING: HF_MODEL_ID is a path, please use HF_MODEL_DIR for paths instead."
    export HF_MODEL_DIR="${HF_MODEL_ID}"
    unset HF_MODEL_ID
fi

# If `HF_MODEL_DIR` is set, then unset it and set `MODEL_DIR` instead
if [[ -n "${HF_MODEL_DIR:-}" ]]; then
    if [[ -z "${MODEL_DIR:-}" ]]; then
        export MODEL_DIR="${HF_MODEL_DIR}"
    else
        echo "WARNING: MODEL_DIR is already set to '${MODEL_DIR}', keeping its value."
    fi
    unset HF_MODEL_DIR
fi

# If `HF_DEFAULT_PIPELINE_NAME` is set, then unset it and set `CUSTOM_HANDLER_FILE` instead
if [[ -n "${HF_DEFAULT_PIPELINE_NAME:-}" ]]; then
    if [[ -z "${CUSTOM_HANDLER_FILE:-}" ]]; then
        export CUSTOM_HANDLER_FILE="${HF_DEFAULT_PIPELINE_NAME}"
    else
        echo "WARNING: CUSTOM_HANDLER_FILE is already set to '${CUSTOM_HANDLER_FILE}', keeping its value."
    fi
    unset HF_DEFAULT_PIPELINE_NAME
fi

# If `MODEL_DIR` is set and is a valid directory
if [[ -n "${MODEL_DIR:-}" ]]; then
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: Provided MODEL_DIR is not a valid directory" >&2
        exit 1
    fi

    # Check if `requirements.txt` exists and if so install dependencies
    if [[ -f "${MODEL_DIR}/requirements.txt" ]]; then
        echo "INFO: Installing custom dependencies from ${MODEL_DIR}/requirements.txt"
        uv pip install --active -r "${MODEL_DIR}/requirements.txt" --no-cache-dir

        # Check if the custom handler file is missing when `requirements.txt` is present
        if [[ ! -f "${MODEL_DIR}/${CUSTOM_HANDLER_FILE}" ]]; then
            echo "WARNING: requirements.txt is present, but ${CUSTOM_HANDLER_FILE} is missing in ${MODEL_DIR}."
            echo "WARNING: If you intend to run custom code, make sure to include ${CUSTOM_HANDLER_FILE}."
        fi
    fi
fi

exec hf-serve "$@"
