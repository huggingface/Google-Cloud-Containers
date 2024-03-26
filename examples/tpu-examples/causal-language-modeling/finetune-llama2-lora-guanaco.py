import argparse

import torch
import torch_xla
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def inference(model, tokenizer):
    prompts = [
        "### Human: From now on, you will act as a nutritionist. I will ask questions about nutrition and you will reply with an explanation on how I can apply it to my daily basis. My first request: What is the main benefit of doing intermittent fastening regularly?",
        "### Human: Was kannst Du im Vergleich zu anderen Large Language Models?",
    ]

    for prompt in prompts:
        device = "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=50
        )  # model.generate only supported on GPU and CPU
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("\n\n")


def train_llama(args):
    raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    model_id = args.model_id

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # Set the padding token to be the same as the end-of-sequence token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=8,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    # Set up the FSDP config. To enable FSDP via SPMD, set xla_fsdp_v2 to True.
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": [
            "LlamaDecoderLayer"  # Specify the layer to wrap according to the model's config
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
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
    )
    # Train the model
    trainer.train()
    trainer.save_model()

    # Inference
    model = AutoModelForCausalLM.from_pretrained(model_id)
    trained_peft_model = PeftModel.from_pretrained(model, "output")
    inference(trained_peft_model, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_llama(args)
