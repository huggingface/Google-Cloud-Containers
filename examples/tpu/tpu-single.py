import torch
import torch_xla

import torch_xla.core.xla_model as xm

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# Set up TPU device.
device = xm.xla_device()
model_id = "google/gemma-2b"

# Load the pretrained model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Load the dataset and format it for training.
data = load_dataset("timdettmers/openassistant-guanaco", split="train")
max_seq_length = 1024

# Finally, set up the trainer and train the model.
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        output_dir="./gemma-2",
        per_device_train_batch_size=1,  # This is actually the global batch size for SPMD.
        max_steps=100,
        logging_steps=1,
        save_steps=25,
        optim="adafactor",
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        dataloader_drop_last=True,  # Required for SPMD.
        # report_to="tensorboard",  # report metrics to tensorboard
    ),
    # peft_config=lora_config, # not tested but should work too
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
)

trainer.train()

trainer.save_model("output")
