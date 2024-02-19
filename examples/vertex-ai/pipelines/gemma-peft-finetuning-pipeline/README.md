# Finetuing LLMs with Vertex AI Pipelines on Google Cloud

With this example, you finetune a pre-trained language model using Vertex AI Pipelines on Google Cloud. 

## Setup Development Environment
1. A Google Cloud Project.
2. [A service account](https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-console) with `Vertex AI User` and `Storage Object Admin` roles for finetuning model.
3. Make sure you have `Python >=3.8`
4. Install the Google Cloud SDK with: `curl https://sdk.cloud.google.com | bash`
5. Authenticate your gcloud sdk with 
```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud auth application-default login
```
6. Install other required libraries with 
```bash
pip install "kfp==2.6.0"  "google-cloud-aiplatform==1.41.0"

```

## Compile the pipeline
```bash
python pipeline.py \
  --compile \
  --compile-pipeline-path "compiled_pipeline.yaml" \
  --model_id "xxxx" \
  --dataset_id "" \
  --learning_rate 1e-4 \
  --train_batch_size 4 \
  --num_epochs 3


```

## Run the pipeline

Before running the pipeline, make sure you have compiled the pipeline. Also, replace the values of `project_id`, `location`, and `vertex_sa` with your own values.

Then run the pipeline with the following command:

```bash
python pipeline.py \
  --run \
  --compile-pipeline-path "compiled_pipeline.yaml" \
  --project_id "gcp-project-id" \
  --location "us-central1" \
  --vertex_sa "service-account-email" \
```

## Troubleshooting

1. You might run into an error when playing with `dataset` as the `format_data` function defined inside the `preprocess_dataset` expects the dataset to have column_names `["question", "answer", "references"]`. You can modify the function to fit your dataset format.

2. You need to define `target_modules` defined inside `LoraConfig` according to the model you are finetuning.

```
peft_config=LoraConfig(
    r=peft_lora_rank,
    lora_alpha=peft_lora_alpha,
    target_modules=[
        'all-linear'
    ],
    bias="none",
    lora_dropout=peft_lora_dropout,
    task_type=TaskType.CAUSAL_LM
)
```

If not defined properly, you might run into an error like `ValueError: Unsupported target_modules: all-linear`