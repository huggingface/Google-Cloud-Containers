# Example to test Gemma fine-tuning on TPU using Hugging Face's Trainer

This example shows how to fine-tune a Gemma model on TPU using Hugging Face's Trainer. We will use cloud TPU v5. It includes samples for single and multi-TPU training. This examples run on a base `v2-alpha-tpuv5-lite` TPU VM without container to demonstrate the usage of Hugging Face's Trainer. 

## Single TPU training

For single TPU training, we will test Gemma 2B model on TPU using Hugging Face's TRL 

1. create TPU VM
```bash
gcloud alpha compute tpus tpu-vm create tpu-gemma-philipp --zone=us-west4-a --accelerator-type=v5litepod-8 --version v2-alpha-tpuv5-lite
```
_Note: Creating an queuing a TPU instance can take some time 5-10 minutes._
2. SSH into the TPU VM
```bash
gcloud alpha compute tpus tpu-vm ssh tpu-gemma-philipp --zone=us-west4-a
```
3. install the required packages
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install typing-extensions
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install transformers datasets accelerate evaluate scikit-learn peft trl tensorboard
```
4. Copy the example script into the TPU VM
```bash
gcloud alpha compute tpus tpu-vm scp tpu-single.py tpu-gemma-philipp:~/ --zone=us-west4-a 
```
5. Set the environment variables
```bash
export HF_TOKEN=xxx # add you token
export PJRT_DEVICE=TPU 
# export XLA_USE_BF16=1
# export XLA_IR_DEBUG=1
# export XLA_HLO_DEBUG=1
# export XLA_USE_SPMD=1
```

6. Run the example script. The example script will fine-tune the Gemma 2B using the `timdettmers/openassistant-guanaco` dataset for 100 steps and saving every 35 steps and at the end. 
```bash
python tpu-single.py
```
It is only a small example to test the TPU training, but the loss should decrease over time. The overall training time should be around 5-10 minutes. It runs for 100 steps and saves every 25 steps.

> [!CAUTION]
> Currently the logging_step is set to 1. If this is increased to a higher number the script "slightly" breaks and get stuck when the model is saved. This is not a known issue and we should investigate this further.


1. Delete the TPU VM
```bash
gcloud alpha compute tpus tpu-vm delete tpu-gemma-philipp --zone=us-west4-a
```



## Multi-TPU training

1. create TPU VM
```bash
gcloud alpha compute tpus tpu-vm create tpu-gemma-philipp --zone=us-west4-a --accelerator-type=v5litepod-8 --version v2-alpha-tpuv5-lite
```
_Note: Creating an queuing a TPU instance can take some time 5-10 minutes._
2. SSH into the TPU VM
```bash
gcloud alpha compute tpus tpu-vm ssh tpu-gemma-philipp --zone=us-west4-a
```
3. install the required packages
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install typing-extensions
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install transformers datasets accelerate evaluate scikit-learn peft trl tensorboard
```
4. Copy the example script into the TPU VM
```bash
wget https://raw.githubusercontent.com/huggingface/transformers/v4.38.1/examples/pytorch/language-modeling/run_clm.py
```
5. create FSDP config file
```bash
echo '{
  "fsdp_transformer_layer_cls_to_wrap": [
    "LlamaDecoderLayer"
  ],
  "xla": true,
  "xla_fsdp_v2": true,
  "xla_fsdp_grad_ckpt": true
}' > fsdp_config.json
```
6. Set the environment variables
```bash
export HF_TOKEN=xxx # add you token
export PJRT_DEVICE=TPU 
export XLA_USE_SPMD=1
export XLA_USE_BF16=1
# export XLA_IR_DEBUG=1
# export XLA_HLO_DEBUG=1
```

7. run llama script
```bash
 python3 run_clm.py --model_name_or_path   meta-llama/Llama-2-7b-hf  --dataset_name timdettmers/openassistant-guanaco      --per_device_train_batch_size 8     --do_train     --output_dir ./test-clm     --block_size 1024     --optim adafactor     --save_steps 50 --max_steps 100     --logging_steps 10 --fsdp "full_shard" --fsdp_config fsdp_config.json --torch_dtype bfloat16 --dataloader_drop_last yes
```
_Note_ 


# Resources

Resources used in this example:
- [Gemma blog](https://huggingface.co/blog/gemma-peft)
- [Example fsdp script](https://huggingface.co/google/gemma-7b/blob/main/examples/example_fsdp.py)
- [TPU v5 training docs](https://cloud.google.com/tpu/docs/v5p-training)


# Known Issues: 
* Currenlty PEFT adapter are not saved, only full model is saved.  
* Saving is not longer working when logging steps > 1.