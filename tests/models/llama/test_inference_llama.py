import os
import unittest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from ..test_model import ModelInferenceTestMixin

torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = "cuda"


class LLaMaInferenceTest(ModelInferenceTestMixin, unittest.TestCase):

    def test_inference(self):
        ckpt = "meta-llama/Llama-2-7b-hf"
        hf_token = token=os.getenv("HF_HUB_READ_TOKEN", None)

        tokenizer = AutoTokenizer.from_pretrained(ckpt, token=hf_token)

        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, token=hf_token)
        model.to(device)

        # To make generation's sample deterministic
        torch.manual_seed(1)

        # Generate
        with torch.no_grad():
            generate_ids = model.generate(inputs.input_ids, max_length=30)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        expected_output = "Hey, are you conscious? Can you talk to me?\nI'm not sure if you can hear me, but I'm talking"

        assert output == expected_output
