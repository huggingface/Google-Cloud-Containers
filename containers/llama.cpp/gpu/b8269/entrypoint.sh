#!/bin/bash

ldconfig 2>/dev/null || echo "WARN: Unable to refresh ld cache, not a big deal in most cases"

# NOTE: We first check that whether the `nvidia-smi` command is found, as
# otherwise it makes no sense to download the artifacts from Google Cloud
# Storage. In this case, we could eventually suggest the user to either use
# an NVIDIA GPU instance on Google Cloud, or rather to use the llama.cpp CPU
# DLC instead.
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi command not found."
    exit 1
fi

# NOTE: Parse the -m / --model argument from the command-line args to check
# whether the model path is a Google Cloud Storage URI (gs://). If so, the
# model is downloaded locally before starting the server, as llama-server
# does not natively support GCS paths. Note that llama-server's argument
# naming differs from the Hugging Face naming convention (e.g. --model vs
# --model-id, --hf-repo vs --model-id, etc.).
ARGS=("$@")
MODEL_VALUE=""
MODEL_ARG_IDX=-1

for i in "${!ARGS[@]}"; do
    arg="${ARGS[$i]}"
    if [[ "$arg" == "-m" || "$arg" == "--model" ]]; then
        MODEL_ARG_IDX=$((i + 1))
        MODEL_VALUE="${ARGS[$MODEL_ARG_IDX]}"
        break
    elif [[ "$arg" == --model=* ]]; then
        MODEL_VALUE="${arg#--model=}"
        MODEL_ARG_IDX=$i
        break
    fi
done

if [[ "$MODEL_VALUE" == gs://* ]]; then
    echo "INFO: Provided -m/--model value '$MODEL_VALUE' is a Google Cloud Storage path."

    TARGET_DIR="/tmp/model"
    mkdir -p "$TARGET_DIR"

    echo "INFO: gcloud storage cp $MODEL_VALUE $TARGET_DIR --recursive"
    gcloud storage cp "$MODEL_VALUE" "$TARGET_DIR" --recursive

    if [ $? -eq 0 ]; then
        echo "INFO: Model downloaded successfully to ${TARGET_DIR}."
        LOCAL_MODEL=$(find "$TARGET_DIR" -type f -name "*.gguf" | head -1)
        if [[ -z "$LOCAL_MODEL" ]]; then
            echo "ERROR: No .gguf file found in ${TARGET_DIR} after download."
            exit 1
        fi
        echo "INFO: Using local model path: $LOCAL_MODEL"
        if [[ "$arg" == --model=* ]]; then
            ARGS[$MODEL_ARG_IDX]="--model=$LOCAL_MODEL"
        else
            ARGS[$MODEL_ARG_IDX]="$LOCAL_MODEL"
        fi
    else
        echo "ERROR: Failed to download model from GCS."
        exit 1
    fi
fi

# NOTE: Given that we need to support CUDA versions earlier than CUDA 13.1,
# we need to include the `cuda-compat` path in `LD_LIBRARY_PATH` when the
# host CUDA version is lower than the one the container was built against;
# whilst we shouldn't include that when CUDA is aligned or higher as
# otherwise it will fail due to it.
if [ -d /usr/local/cuda/compat ]; then
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $3; exit}')

    IFS='.' read -r MAJ MIN PATCH <<EOF
${DRIVER_CUDA:-0.0.0}
EOF
    : "${MIN:=0}"
    : "${PATCH:=0}"

    DRIVER_INT=$((10#${MAJ} * 10000 + 10#${MIN} * 100 + 10#${PATCH}))
    TARGET_INT=$((13 * 10000 + 1 * 100 + 0))

    if [ "$DRIVER_INT" -lt "$TARGET_INT" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"
    fi
fi

exec /app/llama-server "${ARGS[@]}"
