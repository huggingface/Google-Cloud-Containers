# Hugging Face PyTorch Training Containers

The Hugging Face PyTorch Training Containers are Docker containers for training Hugging Face models on Google Cloud AI Platform. There are two containers depending on which accelerator is used, that is GPU and TPU at the moment. The containers come with all the necessary dependencies to train Hugging Face models on Google Cloud AI Platform.

> [!NOTE]
> These containers are named PyTorch containers since PyTorch is the backend framework used for training the models; but it comes with all the required Hugging Face libraries installed.

## Published Containers

To check which of the available Hugging Face DLCs are published, you can either check the [Google Cloud Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) or use the `gcloud` command to list the available containers with the tag `huggingface-pytorch-training` as follows:

```bash
gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io" | grep "huggingface-pytorch-training"
```

## Getting Started

Below you will find the instructions on how to run the PyTorch Training containers available within this repository. Note that before proceeding you need to first ensure that you have Docker installed either on your local or remote instance, if not, please follow the instructions on how to install Docker [here](https://docs.docker.com/get-docker/).

Additionally, if you're willing to run the Docker container in GPUs you will need to install the NVIDIA Container Toolkit.

### Run

The PyTorch Training containers will start a training job that will start on `docker run` and will be closed whenever the training job finishes. As the container is offered for both accelerators GPU and TPU, the examples below are provided.

- **GPU**: This example showcases how to fine-tune an LLM via [`trl`](https://github.com/huggingface/trl) on a GPU instance using the PyTorch Training container, as it comes with `trl` installed.

  ```bash
  docker run --gpus all -ti \
      -v $(pwd)/artifact:/artifact \
      -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310 \
      trl sft \
      --model_name_or_path google/gemma-2b \
      --attn_implementation "flash_attention_2" \
      --torch_dtype "bfloat16" \
      --dataset_name OpenAssistant/oasst_top1_2023-08-25 \
      --dataset_text_field "text" \
      --max_steps 100 \
      --logging_steps 10 \
      --bf16 True \
      --per_device_train_batch_size 4 \
      --use_peft True \
      --load_in_4bit True \
      --output_dir /artifacts
  ```

  > [!NOTE]
  > For a more detailed explanation and a diverse set of examples, please check the [./examples](../../examples) directory that contains examples on both Google Kubernetes Engine (GKE) and Google Vertex AI.

- **TPU**: This example showcases how to deploy a Jupyter Notebook Server from a TPU instance (such as `v5litepod-8`) using the PyTorch Training container, as it comes with `optimum-tpu` installed; so that then you can import a Jupyter Notebook from the ones defined within the `opitimum-tpu` repository or just reuse the Jupyter Notebook that comes within the PyTorch Training container i.e. [`gemma-tuning.ipynb`](https://github.com/huggingface/optimum-tpu/blob/main/examples/language-modeling/gemma_tuning.ipynb); and then just run it.

  ```bash
  docker run --rm --net host --privileged \
      -v$(pwd)/artifact:/notebooks/output \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310 \
      jupyter notebook \
      --port 8888 \
      --allow-root \
      --no-browser \
      notebooks
  ```

  > [!NOTE]
  > Find more detailed examples on TPU fine-tuning in the [`optimum-tpu`](https://github.com/huggingface/optimum-tpu/tree/main/examples) repository.

## Optional

### Build

> [!WARNING]
> Building the containers is not recommended since those are already built by Hugging Face and Google Cloud teams and provided openly, so the recommended approach is to use the pre-built containers available in [Google Cloud's Artifact Registry](https://console.cloud.google.com/artifacts/docker/deeplearning-platform-release/us/gcr.io) instead.

The PyTorch Training containers come with two different containers depending on the accelerator used for training, being either GPU or TPU, those have different constraints when building the Docker image as described below:

- **GPU**: To build the PyTorch Training container for GPU, an instance with at least one NVIDIA GPU available is required to install `flash-attn` (used to speed up the attention layers during training and inference).

  ```bash
  docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310 -f containers/pytorch/training/gpu/2.3.0/transformers/4.42.3/py310/Dockerfile .
  ```

- **TPU**: To build the PyTorch Training container for Google Cloud TPUs, an instance with at least one TPU available is required to install `optimum-tpu` which is a Python library with Google TPU optimizations for `transformers` models, making its integration seamless.

  ```bash
  docker build -t us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310 -f containers/pytorch/training/tpu/2.5.1/transformers/4.46.3/py310/Dockerfile .
  ```

  To run the example notebook for fine-tuning Gemma, use the command below. You can skip the “Environment Setup” step, as you should already be on a TPU-enabled machine. For better security, consider omitting the --allow-root and --NotebookApp.token='' options when running the notebook.

  ```bash
  docker run --rm --net host --privileged \
      -v$(pwd)/artifacts:/tmp/output \
      -e HF_TOKEN=${HF_TOKEN} \
      us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-tpu.2.4.0.transformers.4.41.1.py310 \
      jupyter notebook --allow-root --NotebookApp.token='' /notebooks/gemma_tuning.ipynb
  ```
