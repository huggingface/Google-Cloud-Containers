import argparse

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


def train_gemma(args):
    raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    model_id = "google/gemma-2b"

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = (
            f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        )
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join(
            [i for i in [instruction, context, response] if i is not None]
        )
        sample["text"] = prompt
        return sample

    # apply prompt template
    format_dataset = raw_dataset.map(
        format_dolly, remove_columns=list(raw_dataset.features)
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    device = xm.xla_device()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=8,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    # Set up the FSDP config. To enable FSDP via SPMD, set xla_fsdp_v2 to True.
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": [
            "GemmaDecoderLayer"  # Specify the layer to wrap according to the model's config
        ],
        "xla": True,
        "xla_fsdp_v2": True,
        "xla_fsdp_grad_ckpt": True,
    }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=args.train_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant", 
        dataloader_drop_last=True,  # Required for SPMD.
        fsdp="full_shard",
        fsdp_config=fsdp_config,
    )

    # Initialize our Trainer
    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        args=training_args,
        dataset_text_field="text",
        packing=True,
        train_dataset=format_dataset,
        tokenizer=tokenizer,
    )
    # Train the model
    trainer.train()
    trainer.save_model()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_gemma(args)
