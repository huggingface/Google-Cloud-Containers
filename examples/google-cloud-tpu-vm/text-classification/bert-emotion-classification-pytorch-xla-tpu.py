import argparse

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train_model(args):
    raw_dataset = load_dataset("dair-ai/emotion")
    raw_dataset = raw_dataset.rename_column(
        "label", "labels"
    )  # to match Trainer requirements
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def preprocess_dataset(raw_dataset):
        # Tokenize helper function
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        tokenized_dataset = raw_dataset.map(
            tokenize, batched=True, remove_columns=["text"]
        )
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset

    tokenized_dataset = preprocess_dataset(raw_dataset)

    # Scale learning rate to num cores
    lr = args.lr * xm.xrt_world_size()
    device = xm.xla_device()
    xm.master_print(f"Current TpU: {device}, total-TPU={xm.xrt_world_size()}")

    # Prepare model labels - useful for inference
    labels = tokenized_dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    num_train_steps = int(
        len(tokenized_dataset["train"])
        / args.train_batch_size
        / xm.xrt_world_size()
        * args.num_epochs
    )

    ## Define training arguments
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        learning_rate=lr,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="bert-base-uncased", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_cores", default=8, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    args = parser.parse_args()
    return args


def _mp_fn(index, args):
    torch.set_default_dtype(torch.float32)
    train_model(args)


if __name__ == "__main__":
    args = parse_args()
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)
