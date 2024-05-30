# Hugging Face Pytorch Inference Containers

The Hugging Face Pytorch Inference containers are Python FastAPIs Docker containers for serving 🤗 Transformers models on Google Cloud AI Platform. There are 3 containers, one for CPU, one for GPU, and one for TPU (coming soon). It provides default pre-processing, predict and post-processing for Transformers, Sentence Tranfsformers. It is also possible to define custom handler.py for customizing pre-processing and post-processing steps. The Toolkit is build to work with the Hugging Face Hub.

## Getting Started

### GPU Image

Start by cloning the repository:

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
cd Google-Cloud-Containers
```

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
```

#### Distilbert Test 

Launch the container on a GPU instance (g2) with this command:

```bash
docker run --gpus all -ti -p 8080:8080 -e AIP_MODE=PREDICTION -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/pred -e AIP_HEALTH_ROUTE=/h -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english -e HF_TASK=text-classification us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311
```

Once the Docker container is running, you can send request with the following command:

```bash
curl --request POST \
	--url http://localhost:8080/pred \
	--header 'Content-Type: application/json; charset=UTF-8' \
	--data '{
	"instances": ["I love this product", "I hate this product"],
	"parameters": { "top_k": 2 }
}'
```

For a Vertex AI example checkout [Deploy Distilbert on Vertex AI](../../../examples/vertex-ai/notebooks/deploy-bert-on-vertex-ai.ipynb) notebook.  

### CPU Image

Start by cloning the repository:

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
cd Google-Cloud-Containers
```

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
```

#### Distilbert Test 

Launch the container on a CPU instance with this command:

```bash
docker run -ti -p 8080:8080 -e AIP_MODE=PREDICTION -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/pred -e AIP_HEALTH_ROUTE=/h -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english -e HF_TASK=text-classification us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311
```

Once the Docker container is running, you can send request with the following command:

```bash
curl --request POST \
	--url http://localhost:8080/pred \
	--header 'Content-Type: application/json; charset=UTF-8' \
	--data '{
	"instances": ["I love this product", "I hate this product"],
	"parameters": { "top_k": 2 }
}'
```
