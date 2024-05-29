# Hugging Face Pytorch Inference Container

The Hugging Face Pytorch Inference container is a Python, FastAPI for serving 🤗 Transformers models in containers. This library provides default pre-processing, predict and postprocessing for Transformers, Sentence Tranfsformers. It is also possible to define custom handler.py for customization. The Toolkit is build to work with the Hugging Face Hub.

## Getting Started

### GPU Image

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/gpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
```

#### Distilbert Test 

test the container on a GPU instance (g2) with

```bash
docker run --gpus all -ti -p 8080:8080 -e AIP_MODE=PREDICTION -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/pred -e AIP_HEALTH_ROUTE=/h -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english -e HF_TASK=text-classification us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-gpu.2.2.2.transformers.4.41.1.py311
```

Send request:

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

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311 -f containers/pytorch/inference/cpu/2.2.2/transformers/4.41.1/py311/Dockerfile .
```

#### Distilbert Test 

test the container on a GPU instance (g2) with

```bash
docker run -ti -p 8080:8080 -e AIP_MODE=PREDICTION -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/pred -e AIP_HEALTH_ROUTE=/h -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english -e HF_TASK=text-classification us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2.2.2.transformers.4.41.1.py311
```

Send request:

```bash
curl --request POST \
	--url http://localhost:8080/pred \
	--header 'Content-Type: application/json; charset=UTF-8' \
	--data '{
	"instances": ["I love this product", "I hate this product"],
	"parameters": { "top_k": 2 }
}'
```
