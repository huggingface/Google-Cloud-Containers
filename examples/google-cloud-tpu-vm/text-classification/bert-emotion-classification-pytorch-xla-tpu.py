import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla import runtime as xr
import torch_xla.distributed.parallel_loader as pl
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np


def train_model(args):
    raw_dataset = load_dataset("dair-ai/emotion")
    raw_dataset =  raw_dataset.rename_column("label", "labels") # to match Trainer requirements
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    def preprocess_dataset(raw_dataset):
        # Tokenize helper function
        def tokenize(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=True,return_tensors="pt")
        tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset
    
    tokenized_dataset = preprocess_dataset(raw_dataset)
    train_sampler, test_sampler = None, None
    if xm.xrt_world_size() > 1:
        xm.master_print(f"Training with: {xm.xrt_world_size()} TPU cores") 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset["train"],
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
              tokenized_dataset["test"],
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        ) 
    
    test_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["test"],
        sampler=test_sampler,
        batch_size=args.test_batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        )

    # Scale learning rate to num cores
    lr = args.lr * xm.xrt_world_size()
    device = xm.xla_device()
    
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
    model = model.to(device)
    
    # Synchronize model parameters across replicas manually.
    if xr.using_pjrt():
        xm.broadcast_master_param(model)
    num_train_steps = int(len(tokenized_dataset["train"]) / args.train_batch_size / xm.xrt_world_size() * args.num_epochs)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    optimizer = AdamW(params=model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)
    def single_train_epoch(dataloader):
        model.train()
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer)
            scheduler.step()
            if step % 10 == 0:
                 xm.master_print(f'step={step}, loss={loss}')
    def single_test_epoch(dataloader):
        model.eval()
        total_samples, num_corrects = 0, 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            num_corrects += preds.eq(batch['labels'].view_as(preds)).sum()
            total_samples += batch['labels'].size(0)
        
        acc = 100.0 * num_corrects.item() / total_samples
        acc = xm.mesh_reduce('test_accuracy', acc, np.mean)
        return acc
    
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
    test_device_loader = pl.MpDeviceLoader(test_dataloader, device)
    
    for epoch in range(args.num_epochs):
        xm.master_print(f'Epoch {epoch} training begin')
        single_train_epoch(train_device_loader)
        xm.master_print(f'Epoch {epoch} training end')
        xm.master_print(f'Epoch {epoch} testing begin')
        acc = single_test_epoch(test_device_loader)
        xm.master_print(f'Test-Accuracy: {acc:.2f}% after Epoch {epoch}')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='bert-base-uncased', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_cores', default=8, type=int) 
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()
    return args

def _mp_fn(index, args):
    torch.set_default_dtype(torch.float32)
    train_model(args)
    

if __name__ == '__main__':
    args = parse_args()
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)