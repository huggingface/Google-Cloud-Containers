from kfp import dsl
from kfp.dsl import Output, Dataset, Input, Model
from kfp import compiler
import google.cloud.aiplatform as aip
import argparse

@dsl.component(base_image="python:3.10",
              packages_to_install=["datasets==2.17.0"],
              output_component_file=None)
def download_dataset(dataset_id: str, 
                     raw_train_dataset_artifact: Output[Dataset],
                     raw_val_dataset_artifact: Output[Dataset],
                     raw_test_dataset_artifact: Output[Dataset]
                     ):
    from datasets import load_dataset
    dataset = load_dataset(dataset_id)
    dataset["train"].save_to_disk(raw_train_dataset_artifact.path)
    dataset["validation"].save_to_disk(raw_val_dataset_artifact.path)
    dataset["test"].save_to_disk(raw_test_dataset_artifact.path)


@dsl.component(base_image="python:3.10",
                packages_to_install=["datasets==2.17.0", "transformers==4.37.2"],
                output_component_file=None)
def preprocess_dataset(
    raw_train_dataset_artifact: Input[Dataset],
    raw_val_dataset_artifact: Input[Dataset],
    model_id: str,
    processed_train_dataset_artifact: Output[Dataset],
    processed_val_dataset_artifact: Output[Dataset],
):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    
    def format_data(sample):
        question = sample["question"][0].replace('"', r'\"')
        answer = sample["answer"][0].replace('"', r'\"')
        #unpacking the list of references and creating one string for reference
        references = '\n'.join([f"[{index + 1}] {string}" for index, string in enumerate(sample["references"][0])])
        prompt = f"""###System:
        Read the references provided and answer the corresponding question.
        ###References:
        {references}
        ###Question:
        {question}
        ###Answer:
        {answer}"""
        sample["text"] = prompt
        return sample
    
    raw_train_dataset = load_from_disk(raw_train_dataset_artifact.path)
    raw_validation_dataset = load_from_disk(raw_val_dataset_artifact.path)
    format_train_dataset = raw_train_dataset.map(format_data, remove_columns=list(raw_train_dataset.features))
    format_validation_dataset = raw_validation_dataset.map(format_data, remove_columns=list(raw_validation_dataset.features))
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_train_dataset = format_train_dataset.map(lambda example: tokenizer(example["text"], padding="max_length", truncation=True, max_length=1024), batched=True, remove_columns=format_train_dataset.features)
    tokenized_validation_dataset = format_validation_dataset.map(lambda example: tokenizer(example["text"], padding="max_length", truncation=True, max_length=1024), batched=True, remove_columns=format_validation_dataset.features)
    
    tokenized_train_dataset.save_to_disk(processed_train_dataset_artifact.path)
    tokenized_validation_dataset.save_to_disk(processed_val_dataset_artifact.path)

# ToDo: Add the correct base_image
@dsl.component(base_image="us-central1-docker.pkg.dev/xxx/deep-learning-images/huggingface-pytorch-training-gpu.2.1.transformers.4.37.2.py310:latest")
def train_model(model_id: str, 
                processed_train_data_artifact: Input[Dataset], 
                processed_val_data_artifact: Input[Dataset],
                output_model_artifact: Output[Model],
                warmup_steps: int,
                max_steps: int, 
                learning_rate: float,
                train_batch_size: int,
                eval_batch_size: int,
                gradient_accumulation_steps: int,
                fp16: bool,
                gradient_checkpointing: bool,
                weight_decay: float,
                logging_steps: int,
                logging_strategy: str,
                save_strategy: str,
                save_steps: int,
                eval_strategy: str,
                eval_steps: int,
                optimizer: str,
                peft_lora_rank: int,
                peft_lora_alpha: int,
                peft_lora_dropout: float,
                ):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
    from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, TaskType, get_peft_model
    import torch

    train_dataset = load_from_disk(processed_train_data_artifact.path)
    eval_dataset = load_from_disk(processed_val_data_artifact.path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=bnb_config, torch_dtype=torch.float16)
    
    #gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    # Freeze base model layers and cast layernorm in fp32
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    
    peft_config=LoraConfig(
        r=peft_lora_rank,
        lora_alpha=peft_lora_alpha,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        bias="none",
        lora_dropout=peft_lora_dropout,
        task_type=TaskType.CAUSAL_LM
    )

    peft_model=get_peft_model(model, peft_config)
    training_args=TrainingArguments(
        output_dir=output_model_artifact.path,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=warmup_steps,
        max_steps=max_steps, # Total number of training steps
        learning_rate=learning_rate, # Learning rate
        weight_decay=weight_decay, # Weight decay
        optim=optimizer, # Keep the optimizer state and quantize it
        fp16=fp16, # use fp16 16bit(mixed) precision training instead of 32-bit training.
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
    )

    peft_model.config.use_cache=False

    peft_trainer=Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    peft_trainer.train()

@dsl.pipeline
def finetuning_pipeline(
    dataset_id: str,
    model_id: str, 
    warmup_steps: int,
    max_steps: int, 
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    fp16: bool,
    gradient_checkpointing: bool,
    weight_decay: float,
    logging_steps: int,
    logging_strategy: str,
    save_strategy: str,
    save_steps: int,
    eval_strategy: str,
    eval_steps: int,
    optimizer: str,
    peft_lora_rank: int,
    peft_lora_alpha: int,
    peft_lora_dropout: float,
):
    DatasetDownloadTask = download_dataset(dataset_id=dataset_id).set_display_name("Download Dataset")
    PreprocessTask = preprocess_dataset(raw_train_dataset_artifact=DatasetDownloadTask.outputs["raw_train_dataset_artifact"], 
                                        raw_val_dataset_artifact=DatasetDownloadTask.outputs["raw_val_dataset_artifact"],
                                        model_id=model_id).set_display_name("Preprocess Dataset")
    TrainTask = (train_model(model_id=model_id,
                            processed_train_data_artifact=PreprocessTask.outputs["processed_train_dataset_artifact"],
                            processed_val_data_artifact=PreprocessTask.outputs["processed_val_dataset_artifact"], 
                            warmup_steps=warmup_steps,
                            max_steps=max_steps,
                            learning_rate=learning_rate,
                            train_batch_size=train_batch_size,
                            eval_batch_size=eval_batch_size,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            fp16=fp16,
                            gradient_checkpointing=gradient_checkpointing,
                            weight_decay=weight_decay,
                            logging_steps=logging_steps,
                            logging_strategy=logging_strategy,
                            save_strategy=save_strategy,
                            save_steps=save_steps,
                            eval_strategy=eval_strategy,
                            eval_steps=eval_steps,
                            optimizer=optimizer,
                            peft_lora_rank=peft_lora_rank,
                            peft_lora_alpha=peft_lora_alpha,
                            peft_lora_dropout=peft_lora_dropout)
                            .set_display_name("Train Model")
                            .set_cpu_limit("8")
                            .set_memory_limit("32G")
                            .add_node_selector_constraint("NVIDIA_TESLA_T4")
                            .set_accelerator_limit("1"))

def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("--project_id", type=str, default="gcp-xxxx", help="The project id to use for the pipeline")
    parser.add_argument("--location", type=str, default="us-central1", help="The location to use for the pipeline")
    parser.add_argument("--vertex_sa", type=str, default="xxxx-xxx@developer.gserviceaccount.com")
    parser.add_argument("--dataset_id", type=str, default="THUDM/webglm-qa", help="The dataset id to use for the pipeline")
    parser.add_argument("--model_id", type=str, default="gemma-2b", help="The model id to use for the pipeline") #ToDo: Change and test with Gemma when launched
    parser.add_argument("--warmup_steps", type=int, default=50, help="The warmup steps for the pipeline")
    parser.add_argument("--max_steps", type=int, default=100, help="The max steps for the pipeline")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The learning rate for the pipeline")
    parser.add_argument("--train_batch_size", type=int, default=2, help="The train batch size for the pipeline")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="The eval batch size for the pipeline")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="The gradient accumulation steps for the pipeline")
    parser.add_argument("--fp16", type=bool, default=True, help="The fp16 for the pipeline")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay for the pipeline")
    parser.add_argument("--logging_steps", type=int, default=10, help="The logging steps for the pipeline")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="The logging strategy for the pipeline")
    parser.add_argument("--save_strategy", type=str, default="steps", help="The save strategy for the pipeline")
    parser.add_argument("--save_steps", type=int, default=100, help="The save steps for the pipeline")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="The eval strategy for the pipeline")
    parser.add_argument("--eval_steps", type=int, default=50, help="The eval steps for the pipeline")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="The optimizer for the pipeline")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="The gradient checkpointing for the pipeline")
    parser.add_argument("--peft_lora_rank", type=int, default=16, help="The peft lora rank for the pipeline")
    parser.add_argument("--peft_lora_alpha", type=int, default=32, help="The peft lora alpha for the pipeline")
    parser.add_argument("--peft_lora_dropout", type=float, default=0.05, help="The peft lora dropout for the pipeline")
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    aip.init(project=args.project_id, location=args.location)
    
    # Compile Pipeline
    compiler.Compiler().compile(
        pipeline_func=finetuning_pipeline,
        package_path=f"gs://{args.project_id}-vertex-ai-pipeline/finetuning_pipeline.yaml",
        pipeline_parameters=vars(args),
    )

    # Define and Run Pipeline
    job = aip.PipelineJob(
    display_name="Finetuning-Gemma-CLM",
    template_path=f"gs://{args.project_id}-vertex-ai-pipeline/finetuning_pipeline.yaml",
    pipeline_root=f"gs://{args.project_id}-vertex-ai-pipeline",
    enable_caching=True,
    )

    job.run(args.vertex_sa)

if __name__ == "__main__":
    main()