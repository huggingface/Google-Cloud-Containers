# Finetune Gemma-2B using Hugging Face PyTorch TPU DLC on Google Cloud TPU(v5e)

This example demonstrates how to finetune [gemma-2b](https://huggingface.co/google/gemma-2b) using Hugging Face's DLCs on Google Cloud single-host TPU(v5e) VM. We use the [transformers](https://huggingface.co/docs/transformers/), [TRL](https://huggingface.co/docs/trl/en/index), and [PEFT](https://huggingface.co/docs/peft/index) library to fine-tune. The dataset used for this example is the [Dolly-15k](databricks/databricks-dolly-15k) dataset which can be easily accessed from Hugging Face's [Datasets](https://huggingface.co/datasets) Hub. 


## What are TPUs?

Google Cloud TPUs are custom-designed AI accelerators, which are optimized for training and inference of large AI models. They are ideal for a variety of use cases, such as chatbots, code generation, media content generation, synthetic speech, vision services, recommendation engines, personalization models, among others.

Advantages of using TPUs include:

- Designed to scale cost-efficiently for a wide range of AI workloads, spanning training, fine-tuning, and inference.
- Optimized for TensorFlow, PyTorch, and JAX, and are available in a variety of form factors, including edge devices, workstations, and cloud-based infrastructure.
- TPUs are available in [Google Cloud](https://cloud.google.com/tpu/docs/intro-to-tpu), and have been integrated with [Vertex AI](https://cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm), and [Google Kubernetes Engine (GKE)](https://cloud.google.com/tpu?hl=en#cloud-tpu-in-gke).
- 

## Before you begin

Make sure you have the following:
- A Google Cloud project with billing enabled.
<!-- - Access to Hugging Face's PyTorch TPU DLC. -->
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install#linux) installed on your local machine.

For installing Google Cloud CLI, you can use the following commands:

```bash
curl https://sdk.cloud.google.com | bash
exec zsh -l
gcloud init
```

You can configure your Google Cloud project using the following command:

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud auth application-default login
```

Enable the Compute Engine and Cloud TPU APIs using the following commands:

```bash
gcloud services enable compute.googleapis.com
gcloud services enable tpu.googleapis.com
```


## Spin up a TPU VM on Google Cloud

We will be using [Cloud TPU v5e](https://cloud.google.com/tpu/docs/v5e-training), Google Cloud's latest generation AI accelerator. We will setup a single-host TPU(v5e) VM to train the model. 

You can read more about Single-host(8 chips) and Multi-host(> 8 chips) TPU VMs on [Google Cloud TPU configurations](https://cloud.google.com/tpu/docs/supported-tpu-configurations).

Note: Steps to run the example would differ for multi-host TPU VMs. One would need to use [SAX](https://github.com/google/saxml) for multi-host training and multi-host inference.

To [set up a TPU VM](https://cloud.google.com/tpu/docs/setup-gcp-account#set-up-env), follow the steps below:

<!-- TODO: Update this script to directly use the Hugging Face PyTorch TPU DLC -->

```bash
gcloud alpha compute tpus tpu-vm create dev-tpu-vm \
--zone=us-west4-a \
--accelerator-type=v5litepod-8 \
--version v2-alpha-tpuv5-lite
```

After some time, the TPU VM will be created. You can see the list of TPU VMs in [Google Cloud console](https://console.cloud.google.com/compute/tpus).


## Set up the environment

Once, the Cloud TPU VM is up and running, you can SSH into the VM using the following command:

```bash
gcloud alpha compute tpus tpu-vm ssh dev-tpu-vm --zone=us-west4-a
```

<!-- TODO: Update the link to the Dockerfile and remove the part where docker image needs to be build once DLCs are released-->
You now need to build the environment using Hugging Face's PyTorch TPU DLC [Dockerfile](https://github.com/huggingface/Google-Cloud-Containers/blob/feature/pytorch-tpu-container/containers/pytorch/training/tpu/2.1/transformers/4.37.2/py310/Dockerfile). You can use the following commands to build the environment:

```bash
git clone https://github.com/huggingface/Google-Cloud-Containers.git
cd Google-Cloud-Containers
sudo docker build -t huggingface-pytorch-training-tpu-2.3.transformers.4.38.1.py310:latest -f containers/pytorch/training/tpu/2.3/transformers/4.38.1/py310/Dockerfile .
```

## Train the model
Once, the docker image is built, we need to run the docker container in order to activate the enviroment. You can use the following commands to run the docker container:

```bash
sudo docker run -it -v $(pwd):/workspace --privileged huggingface-pytorch-training-tpu-2.3.transformers.4.38.1.py310:latest bash
```

Now, you can run the following commands to train the model:

```bash
export PJRT_DEVICE=TPU XLA_USE_BF16=1 XLA_USE_SPMD=1
export HF_TOKEN=<YOUR_HF_TOKEN>
cd /workspace
python examples/google-cloud-tpu-vm/causal-language-modeling/finetune-gemma-lora-dolly.py \ 
--num_epochs 3 \
--train_batch_size 16 \
--lr 3e-4
```