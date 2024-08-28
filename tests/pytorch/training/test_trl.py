import os
import pytest
import subprocess

from pathlib import PosixPath
from transformers import AutoModelForCausalLM

from tests.constants import CUDA_AVAILABLE


MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_trl(tmp_path: PosixPath) -> None:
    """Adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py"""
    # Set `TRL_USE_CLI` to `0` to avoid using `rich` in the CLI
    test_env = os.environ.copy()
    test_env["TRL_USE_RICH"] = "0"

    subprocess.run(
        [
            "trl",
            "sft",
            f"--model_name_or_path={MODEL_ID}",
            "--dataset_text_field=text",
            "--report_to=none",
            "--learning_rate=1e-5",
            "--per_device_train_batch_size=8",
            "--gradient_accumulation_steps=1",
            f"--output_dir={str(tmp_path / 'sft_openassistant-guanaco')}",
            "--logging_steps=1",
            "--num_train_epochs=-1",
            "--max_steps=10",
            "--gradient_checkpointing",
        ],
        env=test_env,
        check=True,
    )

    assert (tmp_path / "sft_openassistant-guanaco").exists()
    assert (tmp_path / "sft_openassistant-guanaco" / "model.safetensors").exists()

    _ = AutoModelForCausalLM.from_pretrained(
        (tmp_path / "sft_openassistant-guanaco").as_posix()
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_trl_peft(tmp_path: PosixPath) -> None:
    """Adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py"""
    # Set `TRL_USE_CLI` to `0` to avoid using `rich` in the CLI
    test_env = os.environ.copy()
    test_env["TRL_USE_RICH"] = "0"

    subprocess.run(
        [
            "trl",
            "sft",
            f"--model_name_or_path={MODEL_ID}",
            "--dataset_text_field=text",
            "--report_to=none",
            "--learning_rate=1e-5",
            "--per_device_train_batch_size=8",
            "--gradient_accumulation_steps=1",
            f"--output_dir={str(tmp_path / 'sft_openassistant-guanaco')}",
            "--logging_steps=1",
            "--num_train_epochs=-1",
            "--max_steps=10",
            "--gradient_checkpointing",
            "--use_peft",
            "--lora_r=64",
            "--lora_alpha=16",
        ],
        env=test_env,
        check=True,
    )

    assert (tmp_path / "sft_openassistant-guanaco").exists()
    assert (tmp_path / "sft_openassistant-guanaco" / "adapter_config.json").exists()
    assert (
        tmp_path / "sft_openassistant-guanaco" / "adapter_model.safetensors"
    ).exists()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.load_adapter((tmp_path / "sft_openassistant-guanaco").as_posix())
