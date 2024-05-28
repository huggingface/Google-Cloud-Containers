# Hugging Face Pytorch Trainning Container

The Hugging Face Pytorch Trainning Container is a Docker container for training Hugging Face models on Google Cloud AI Platform. The container comes with all the necessary dependencies to train Hugging Face models on Google Cloud AI Platform.

## Build Image 

Build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310 -f containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile .
```

## Test & Fine-tune Gemma using TRL

The command below shows how to test and fine-tune Gemma using TRL. First we need to login into Hugging Face to access the gated model. 

```bash
huggingface-cli login --token YOUR_TOKEN
``` 

Then we can run the following command to test and fine-tune Gemma using TRL and Q-Lora on 100 steps with flash attention.

_NOTE: Parameters are tuned for a GPU with 24GB._ 

```bash
docker run --gpus all -ti -v $(pwd)/artifcats:/artifacts -e HF_TOKEN=$(cat ~/.cache/huggingface/token) us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310 \
trl sft \
--model_name_or_path google/gemma-2b \
--attn_implementation "flash_attention_2"  \
--torch_dtype "bfloat16"  \
--dataset_name OpenAssistant/oasst_top1_2023-08-25  \
--dataset_text_field "text"  \
--max_steps 100  \
--logging_steps 10  \
--bf16 True  \
--per_device_train_batch_size 4  \
--use_peft True  \
--load_in_4bit True  \
--output_dir /artifacts
```

This will now fine-tune Gemma on the OpenAssistant dataset using the `text` column and provided CLI Parameters. Learn more about the [TRL CLI HERE](https://huggingface.co/docs/trl/clis). Alteratively we support yaml configuration files. See [gemma-2b-test.yaml](gemma-2b-test.yaml).

```bash
docker run --gpus all -ti -v $(pwd)/artifcats:/artifacts -v $(pwd)/containers/pytorch/training/gemma-2b-test.yaml:/config/gemma-2b-test.yaml -e HF_TOKEN=$(cat ~/.cache/huggingface/token) us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310 \
trl sft --config /config/gemma-2b-test.yaml
```

_NOTE: This should make the integration into Vertex AI seamless._


For a Vertex AI example checkout [Fine-Tune Gemma](TODO:) notebook.  
