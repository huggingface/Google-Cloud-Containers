from kfp import dsl
from kfp.dsl import Output, Dataset, Input, Model
from kfp import compiler
import google.cloud.aiplatform as aip
import argparse
import os

# ToDo: Add the correct base_image
@dsl.component(base_image="us-central1-docker.pkg.dev/xxxx/deep-learning-images/huggingface-pytorch-training-gpu.2.1.transformers.4.37.2.py310:latest",
              packages_to_install=[],
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

# ToDo: Add the correct base_image
@dsl.component(base_image="us-central1-docker.pkg.dev/xxxx/deep-learning-images/huggingface-pytorch-training-gpu.2.1.transformers.4.37.2.py310:latest",
                packages_to_install=[],
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
    
    ## Check if the dataset has the columns that format_data function expects
    expected_column_names = ["question", "answer", "references"]
    for column_name in expected_column_names:
        if column_name not in raw_train_dataset.column_names:
            raise ValueError(f"Column {column_name} not found in the dataset, please write your own format_data function to handle the dataset.")
    
    format_train_dataset = raw_train_dataset.map(format_data, remove_columns=list(raw_train_dataset.features))
    format_validation_dataset = raw_validation_dataset.map(format_data, remove_columns=list(raw_validation_dataset.features))
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_train_dataset = format_train_dataset.map(lambda example: tokenizer(example["text"], padding="max_length", truncation=True), batched=True, remove_columns=format_train_dataset.features)
    tokenized_validation_dataset = format_validation_dataset.map(lambda example: tokenizer(example["text"], padding="max_length", truncation=True), batched=True, remove_columns=format_validation_dataset.features)
    
    tokenized_train_dataset.save_to_disk(processed_train_dataset_artifact.path)
    tokenized_validation_dataset.save_to_disk(processed_val_dataset_artifact.path)

# ToDo: Add the correct base_image
@dsl.component(base_image="us-central1-docker.pkg.dev/xxxxx/deep-learning-images/huggingface-pytorch-training-gpu.2.1.transformers.4.37.2.py310:latest")
def train_model(model_id: str, 
                processed_train_data_artifact: Input[Dataset], 
                processed_val_data_artifact: Input[Dataset],
                output_model_artifact: Output[Model],
                epochs: int,
                learning_rate: float,
                train_batch_size: int,
                eval_batch_size: int,
                gradient_accumulation_steps: int,
                fp16: bool,
                gradient_checkpointing: bool,
                weight_decay: float,
                logging_strategy: str,
                save_strategy: str,
                eval_strategy: str,
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_type=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 device_map='auto', 
                                                 quantization_config=bnb_config,
                                                 use_cache=False, 
                                                 torch_dtype=torch.float16)
    
    # Freeze base model layers and cast layernorm in fp32
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    
    peft_config=LoraConfig(
        r=peft_lora_rank,
        lora_alpha=peft_lora_alpha,
        target_modules=[
            'q-proj', 'v-proj', 'k-proj'
        ],
        bias="none",
        lora_dropout=peft_lora_dropout,
        task_type=TaskType.CAUSAL_LM
    )

    peft_model=get_peft_model(model, peft_config)
    training_args=TrainingArguments(
        output_dir=output_model_artifact.path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        weight_decay=weight_decay, 
        optim=optimizer, 
        fp16=fp16,
        logging_strategy=logging_strategy,
        save_strategy=save_strategy,
        evaluation_strategy=eval_strategy,
        load_best_model_at_end=True,
    )

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
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    fp16: bool,
    gradient_checkpointing: bool,
    weight_decay: float,
    logging_strategy: str,
    save_strategy: str,
    eval_strategy: str,
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
                            epochs=epochs, 
                            learning_rate=learning_rate,
                            train_batch_size=train_batch_size,
                            eval_batch_size=eval_batch_size,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            fp16=fp16,
                            gradient_checkpointing=gradient_checkpointing,
                            weight_decay=weight_decay,
                            logging_strategy=logging_strategy,
                            save_strategy=save_strategy,
                            eval_strategy=eval_strategy,
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
    parser.add_argument("--project_id", type=str, default="xxx", help="The project id to use for training the model")
    parser.add_argument("--location", type=str, default="us-central1", help="The location to use for training the model")
    parser.add_argument("--vertex_sa", type=str, default="xxxxx", help="The service account to use for training the model")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the pipeline")
    parser.add_argument("--run", action="store_true", help="Whether to run the pipeline")
    parser.add_argument("--compile_pipeline_path", type=str, default="finetuning_pipeline.yaml", help="The path to save the compiled pipeline to")
    parser.add_argument("--dataset_id", type=str, default="THUDM/webglm-qa", help="The dataset id to use for training the model")
    parser.add_argument("--model_id", type=str, default="gemma-2b", help="The model id to use for training") #ToDo: Change and test with Gemma when launched
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train the model for")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=2, help="The batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="The batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--fp16", type=bool, default=True, help=" Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay for training the model")
    parser.add_argument("--logging_strategy", type=str, default="epochs", help="The logging strategy to adopt during training")
    parser.add_argument("--save_strategy", type=str, default="epochs", help="The checkpoint save strategy to adopt during training.")
    parser.add_argument("--eval_strategy", type=str, default="epochs", help="The evaluation strategy to adopt during training")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit", help="Optimizer to use for training")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing for training the model")
    parser.add_argument("--peft_lora_rank", type=int, default=16, help=" Lora attention dimension aka the rank.")
    parser.add_argument("--peft_lora_alpha", type=int, default=32, help="The alpha parameter for Lora scaling.")
    parser.add_argument("--peft_lora_dropout", type=float, default=0.05, help="the dropout probability for Lora layers.")    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.compile:
        excluded_args = ['project_id', 'location', 'vertex_sa', 'compile', 'run', 'compile_pipeline_path']
        pipeline_parameters = {key: value for key, value in vars(args).items() if key not in excluded_args}
        
        # Compile Pipeline
        compiler.Compiler().compile(
            pipeline_func=finetuning_pipeline,
            package_path=args.compile_pipeline_path,
            pipeline_parameters=pipeline_parameters,
        )

    if args.run:
        # Define and Run Pipeline
        if not os.path.exists(args.compile_pipeline_path):
            raise ValueError(f"You must compile the pipeline before running it.")
        job = aip.PipelineJob(
        display_name="Finetuning-Gemma-CLM",
        template_path=args.compile_pipeline_path,
        pipeline_root=f"gs://{args.project_id}-vertex-pipelines-{args.location}",
        enable_caching=True,
        )

        job.run(service_account=args.vertex_sa)

if __name__ == "__main__":
    main()