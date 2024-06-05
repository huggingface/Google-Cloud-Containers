# Hugging Face Pytorch Training Containers

The Hugging Face Pytorch Training Containers are Docker containers for training Hugging Face models on Google Cloud AI Platform. There are 2 containers, one for GPU and one for TPU. The containers come with all the necessary dependencies to train Hugging Face models on Google Cloud AI Platform.

## Getting Started

### Build GPU Image Manually

Start by cloning the repository:

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
cd Google-Cloud-Containers
```

Then, build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310 -f containers/pytorch/training/gpu/2.3.0/transformers/4.41.1/py310/Dockerfile .
```

#### Test & Fine-tune Gemma Using TRL

The command below shows how to test and fine-tune Gemma using TRL. First we need to login into Hugging Face to access the gated model. 

```bash
huggingface-cli login --token YOUR_TOKEN
``` 

Once connected to the instance of your choice to use the Hugging Face Pytorch Training Container for GPU, run the following command to test the container and fine-tune Gemma using TRL and Q-Lora on 100 steps with flash attention. This will now fine-tune Gemma on the OpenAssistant dataset using the `text` column and provided CLI Parameters. Learn more about the [TRL CLI HERE](https://huggingface.co/docs/trl/clis).

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

Alteratively we support yaml configuration files. See [gemma-2b-test.yaml](gemma-2b-test.yaml).

```bash
docker run --gpus all -ti -v $(pwd)/artifacts:/artifacts -v $(pwd)/containers/pytorch/training/gemma-2b-test.yaml:/config/gemma-2b-test.yaml -e HF_TOKEN=$(cat ~/.cache/huggingface/token) us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-gpu.2.3.0.transformers.4.41.1.py310 \
trl sft --config /config/gemma-2b-test.yaml
```

_NOTE: This should make the integration into Vertex AI seamless._


For a Vertex AI example checkout [Fine-Tune Gemma](TODO:) notebook.  

### Build TPU Image Manually

Start by cloning the repository:

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers
cd Google-Cloud-Containers
```

Then, build the container with the following command:

```bash
docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310 -f containers/pytorch/training/tpu/2.4.0/transformers/4.41.1/py310/Dockerfile .
```

#### Test & Fine-tune Gemma Using Optimum TPU

There is a [Jupyter notebook](https://github.com/huggingface/optimum-tpu/blob/v0.1.0a0/examples/language-modeling/gemma_tuning.ipynb) explaining how to fine-tune `gemma-2b` model that can be run on a `v5litepod-8` instance using `optimum-tpu`. For convenience, the 
After you created the TPU VM instance, you can use `ssh` or `gcloud` to log in and forward the port `8888`, e.g.:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        -- -L 8888:localhost:8888
```

Once you have access to the TPU VM, launch the container on a Optimum TPU instance:

```bash
docker run --rm --net host --privileged -v$(pwd)/artifacts:/notebooks/output us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310 jupyter notebook --port 8888  --allow-root --no-browser notebooks
```

The Jupyter output will show you an address accessible from your browser, similar to this one:

```
http://localhost:8888/tree?token=3cccbfa5a066217bbb38dc088ca06219e8f2330741b4135c
```

You can then click on the `gemma_tuning.ipynb` and walk through the steps to train `gemma`. To terminate the execution, you can just type `ctrl+c` on the terminal.
