import os
from google.cloud import aiplatform

if __name__ == "__main__":
    PROJECT_ID = "my-project"
    LOCATION = "my-location"
    BUCKET_URI = "gs://hf-vertex-pipelines-eu"

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_URI,
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name="trl-lora-sft",
        container_uri="...",
        command=[
            "sh",
            "-c",
            " && ".join(
                (
                    # required since there's a bug with the `torch_dtype` that prevents us from loading
                    # the model as the default is fp32 and that won't fit in an L4 GPU with 24GiB
                    # see https://github.com/huggingface/trl/issues/1751
                    'pip install "trl @ git+https://github.com/alvarobartt/trl.git@main" --upgrade',
                    "pip install flash-attn --no-build-isolation",
                    'exec trl sft "$@"',
                )
            ),
            "--",
        ],
    )

    args = [
        # MODEL
        "--model_name_or_path=mistralai/Mistral-7B-v0.3",
        "--torch_dtype=bfloat16",
        "--attn_implementation=flash_attention_2",
        # DATASET
        "--dataset_name=timdettmers/openassistant-guanaco",
        "--dataset_text_field=text",
        # PEFT
        "--use_peft",
        "--lora_r=16",
        "--lora_alpha=32",
        "--lora_dropout=0.1",
        "--lora_target_modules=all-linear",
        # TRAINER
        "--bf16",
        "--max_seq_length=1024",
        "--per_device_train_batch_size=2",
        "--gradient_accumulation_steps=8",
        "--gradient_checkpointing",
        "--learning_rate=0.0002",
        "--lr_scheduler_type=cosine",
        "--optim=adamw_bnb_8bit",
        "--num_train_epochs=1",
        "--logging_steps=10",
        "--do_eval",
        "--eval_steps=100",
        "--report_to=none",
        f"--output_dir={BUCKET_URI.replace('gs://', '/gcs/')}/Mistral-7B-v0.3-LoRA-SFT-Guanaco",
        "--overwrite_output_dir",
        "--seed=42",
        "--log_level=info",
    ]

    job.submit(
        args=args,
        replica_count=1,
        machine_type="g2-standard-12",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        base_output_dir=f"{BUCKET_URI}/Mistral-7B-v0.3-LoRA-SFT-Guanaco",
        environment_variables={
            "HF_TOKEN": os.getenv("HF_TOKEN", None),
            "ACCELERATE_LOG_LEVEL": "INFO",
            "TRANSFORMERS_LOG_LEVEL": "INFO",
        },
        timeout=60 * 60 * 3,  # 3 hours (10800s)
        create_request_timeout=60 * 10,  # 10 minutes (600s)
    )
