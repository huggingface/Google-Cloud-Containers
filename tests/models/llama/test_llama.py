import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer


raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
model_id = "meta-llama/Llama-2-7b-hf"


# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    # num_train_epochs=1,
    max_steps=100,
    logging_strategy="steps",
    logging_steps=20,
    bf16=True,
    optim="paged_adamw_8bit",

)

# Initialize our Trainer
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=training_args,
    dataset_text_field="text",
    packing=True,
    train_dataset=raw_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the model
trainer.train()
