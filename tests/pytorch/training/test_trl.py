import logging
import os
import pytest

import docker
from docker.types.containers import DeviceRequest
from pathlib import PosixPath
from transformers import AutoModelForCausalLM

from tests.constants import CUDA_AVAILABLE


MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_trl(caplog: pytest.LogCaptureFixture, tmp_path: PosixPath) -> None:
    """Adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py"""
    caplog.set_level(logging.INFO)

    client = docker.from_env()

    logging.info("Running the container for TRL...")
    container = client.containers.run(
        os.getenv(
            "TRAINING_DLC",
            "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-2.transformers.4-44.ubuntu2204.py311",
        ),
        cmd=[
            "trl",
            "sft",
            f"--model_name_or_path={MODEL_ID}",
            "--dataset_text_field=text",
            "--report_to=none",
            "--learning_rate=1e-5",
            "--per_device_train_batch_size=8",
            "--gradient_accumulation_steps=1",
            "--output_dir=/opt/huggingface/trained_model",
            "--logging_steps=1",
            "--num_train_epochs=-1",
            "--max_steps=10",
            "--gradient_checkpointing",
        ],
        environment={
            "TRL_USE_RICH": 0,
            "ACCELERATE_LOG_LEVEL": "INFO",
            "TRANSFORMERS_LOG_LEVEL": "INFO",
            "TQDM_POSITION": -1,
        },
        platform="linux/amd64",
        # To show all the `logging` messages from the container
        stdin_open=True,
        tty=True,
        # Mount the volume from the `tmp_path` to the `/opt/huggingface/trained_model`
        volumes={
            f"{tmp_path}/sft_openassistant-guanaco": {
                "bind": "/opt/huggingface/trained_model",
                "mode": "rw",
            }
        },
        # Extra kwargs related to the CUDA devices
        runtime="nvidia",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    )

    assert (tmp_path / "sft_openassistant-guanaco").exists()
    assert (tmp_path / "sft_openassistant-guanaco" / "model.safetensors").exists()

    _ = AutoModelForCausalLM.from_pretrained(
        (tmp_path / "sft_openassistant-guanaco").as_posix()
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_trl_peft(caplog: pytest.LogCaptureFixture, tmp_path: PosixPath) -> None:
    """Adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py"""
    caplog.set_level(logging.INFO)

    client = docker.from_env()

    logging.info("Running the container for TRL...")
    container = client.containers.run(
        os.getenv(
            "TRAINING_DLC",
            "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-2.transformers.4-44.ubuntu2204.py311",
        ),
        cmd=[
            "trl",
            "sft",
            f"--model_name_or_path={MODEL_ID}",
            "--dataset_text_field=text",
            "--report_to=none",
            "--learning_rate=1e-5",
            "--per_device_train_batch_size=8",
            "--gradient_accumulation_steps=1",
            "--output_dir=/opt/huggingface/trained_model",
            "--logging_steps=1",
            "--num_train_epochs=-1",
            "--max_steps=10",
            "--gradient_checkpointing",
            "--use_peft",
            "--lora_r=64",
            "--lora_alpha=16",
        ],
        environment={
            "TRL_USE_RICH": 0,
            "ACCELERATE_LOG_LEVEL": "INFO",
            "TRANSFORMERS_LOG_LEVEL": "INFO",
            "TQDM_POSITION": -1,
        },
        platform="linux/amd64",
        # To show all the `logging` messages from the container
        stdin_open=True,
        tty=True,
        # Mount the volume from the `tmp_path` to the `/opt/huggingface/trained_model`
        volumes={
            f"{tmp_path}/sft_openassistant-guanaco": {
                "bind": "/opt/huggingface/trained_model",
                "mode": "rw",
            }
        },
        # Extra kwargs related to the CUDA devices
        runtime="nvidia",
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    )

    assert (tmp_path / "sft_openassistant-guanaco").exists()
    assert (tmp_path / "sft_openassistant-guanaco" / "adapter_config.json").exists()
    assert (
        tmp_path / "sft_openassistant-guanaco" / "adapter_model.safetensors"
    ).exists()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.load_adapter((tmp_path / "sft_openassistant-guanaco").as_posix())
