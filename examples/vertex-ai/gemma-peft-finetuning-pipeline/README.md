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

## Run the pipeline
```bash
python pipeline.py \
  --project_id "gcp-project-id" \
  --location "us-central1" \
  --vertex_sa "service-account-email" \
  --model_id "xxxx" \
  --learning_rate 1e-4 \
  --train_batch_size 4 \
  --num_epochs 3
```