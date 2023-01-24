import os

import torch
import transformers
import wandb
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BloomTokenizerFast, get_scheduler

from petals import DistributedBloomForCausalLM

def main():
    dataset = load_dataset("bavard/personachat_truecased")

    MODEL_NAME = "bigscience/bloomz-petals"
    NUM_PREFIX_TOKENS = 1
    DEVICE = 'cuda'
    BATCH_SIZE = 1
    LR = 1e-2
    WEIGHT_DECAY = 0.0
    NUM_SAMPLES = 1000
    SEED = 42
    MODEL_MAX_LENGTH = 256
    TUNING_MODE = 'ptune'

    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME,
        pre_seq_len=NUM_PREFIX_TOKENS, 
        tuning_mode=TUNING_MODE,
        startup_timeout=120
    ).to(DEVICE)

    def chunking(examples):
        inputs = [
            "\n-----\n".join(history) + "\n-----\n" + candidate
            for history, candidates in zip(examples["history"], examples["candidates"])
            for candidate in candidates
        ]
        return {"chunks": inputs}


    def tokenize(examples):
        outputs = {
            "input_ids": tokenizer(examples["chunks"], padding='max_length', truncation=True)["input_ids"]
        }
        outputs["labels"] = outputs["input_ids"]
        return outputs

    tokenized_datasets = (
        dataset
            .map(chunking, batched=True, remove_columns=dataset["train"].column_names)
            .map(tokenize, batched=True, remove_columns=["chunks"])
    )


    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    train_dataloader = DataLoader(
        train_dataset.select(list(range(NUM_SAMPLES))),
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
    )

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.requires_grad, p.device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)
    )

    wandb.init(
        project="bloom-personachat",
        config={
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_prefix_tokens": NUM_PREFIX_TOKENS,
            "model_name": MODEL_NAME,
            "seed": SEED,

        }
    )

    for batch in tqdm(train_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        model.train()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        wandb.log({"Train Loss": loss})

if __name__ == "__main__":
    print("hello")
    main()

