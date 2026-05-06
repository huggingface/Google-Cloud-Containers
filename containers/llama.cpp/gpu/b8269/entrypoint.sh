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

# NOTE: Parse -m / --model and --mmproj arguments from the command-line args
# to check whether the paths are Google Cloud Storage URIs (gs://). If so,
# the files are downloaded locally before starting the server, as llama-server
# does not natively support GCS paths. Note that llama-server's argument
# naming differs from the Hugging Face naming convention (e.g. --model vs
# --model-id, --hf-repo vs --model-id, etc.).

# Downloads a gs:// path to a local directory and rewrites the arg in ARGS.
# Usage: download_gcs_arg <arg_name_display> <target_dir> <arg_idx> <equals_form>
download_gcs_arg() {
    local display_name="$1"
    local target_dir="$2"
    local arg_idx="$3"
    local equals_form="$4" # 1 if --flag=value form, 0 if --flag value form
    local gcs_uri="${ARGS[$arg_idx]}"

    # Strip the flag prefix when in --flag=value form to get the raw URI
    if [[ "$equals_form" == "1" ]]; then
        gcs_uri="${gcs_uri#*=}"
    fi

    echo "INFO: Provided $display_name value '$gcs_uri' is a Google Cloud Storage path."
    mkdir -p "$target_dir"

    echo "INFO: gcloud storage cp $gcs_uri $target_dir --recursive"
    gcloud storage cp "$gcs_uri" "$target_dir" --recursive

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download $display_name from GCS."
        exit 1
    fi

    echo "INFO: $display_name downloaded successfully to ${target_dir}."
    local local_path
    local gcs_basename
    gcs_basename=$(basename "$gcs_uri")
    if [[ "$gcs_basename" == *.* ]]; then
        local_path="${target_dir}/${gcs_basename}"
        if [[ ! -f "$local_path" ]]; then
            echo "ERROR: Downloaded file not found at ${local_path}."
            exit 1
        fi
    else
        local_path="$target_dir"
    fi
    echo "INFO: Using local path: $local_path"

    if [[ "$equals_form" == "1" ]]; then
        local flag="${ARGS[$arg_idx]%%=*}"
        ARGS[$arg_idx]="${flag}=${local_path}"
    else
        ARGS[$arg_idx]="$local_path"
    fi
}

ARGS=("$@")

for i in "${!ARGS[@]}"; do
    arg="${ARGS[$i]}"
    if [[ "$arg" == "-m" || "$arg" == "--model" ]]; then
        value_idx=$((i + 1))
        if [[ "${ARGS[$value_idx]}" == gs://* ]]; then
            download_gcs_arg "-m/--model" "/tmp/model" "$value_idx" "0"
        fi
    elif [[ "$arg" == --model=* ]]; then
        if [[ "${arg#--model=}" == gs://* ]]; then
            download_gcs_arg "--model" "/tmp/model" "$i" "1"
        fi
    elif [[ "$arg" == "--mmproj" ]]; then
        value_idx=$((i + 1))
        if [[ "${ARGS[$value_idx]}" == gs://* ]]; then
            download_gcs_arg "--mmproj" "/tmp/model" "$value_idx" "0"
        fi
    elif [[ "$arg" == --mmproj=* ]]; then
        if [[ "${arg#--mmproj=}" == gs://* ]]; then
            download_gcs_arg "--mmproj" "/tmp/model" "$i" "1"
        fi
    fi
done

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
