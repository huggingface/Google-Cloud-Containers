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

# NOTE: Given that we need to support CUDA versions earlier than CUDA 12.9.1, we
# need to include the `cuda-compat-12-9` in `LD_LIBRARY_PATH` when the host CUDA
# version is lower than that; whilst we shouldn't include that when CUDA is 13.0+
# as otherwise it will fail due to it.
if [ -d /usr/local/cuda/compat ]; then
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $3; exit}')

    IFS='.' read -r MAJ MIN PATCH <<EOF
${DRIVER_CUDA:-0.0.0}
EOF
    : "${MIN:=0}"
    : "${PATCH:=0}"

    DRIVER_INT=$((10#${MAJ} * 10000 + 10#${MIN} * 100 + 10#${PATCH}))
    TARGET_INT=$((12 * 10000 + 9 * 100 + 1))

    if [ "$DRIVER_INT" -lt "$TARGET_INT" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"
    fi
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 "$@"
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec text-embeddings-router-80 "$@"
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 "$@"
elif [ ${compute_cap} -eq 100 ]; then
    exec text-embeddings-router-100 "$@"
elif [ ${compute_cap} -eq 120 ]; then
    exec text-embeddings-router-120 "$@"
else
    echo "ERROR: CUDA compute capability ${compute_cap} is not supported"
    exit 1
fi
