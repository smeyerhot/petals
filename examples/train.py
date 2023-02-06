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
    # os.environ['HIVEMIND_LOGLEVEL'] = "DEBUG"
    # os.environ['GOLOG_LOG_LEVEL'] = "DEBUG"

    # MODEL_NAME = "./model_save"
    MODEL_NAME = "bigscience/bloomz-petals"
    TOK_NAME = "bigscience/bloomz-petals"
    TUNING_MODE = 'ptune'

    TUNING_MODE = 'ptune'

    NUM_PREFIX_TOKENS = 16
    DEVICE = 'cuda'
    BATCH_SIZE = 8
    LR = 1e-2
    WEIGHT_DECAY = 0.0
    NUM_SAMPLES = 1000
    SEED = 42
    MODEL_MAX_LENGTH = 256
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME,
        pre_seq_len=NUM_PREFIX_TOKENS, 
        tuning_mode=TUNING_MODE,
        request_timeout=300
    ).to(DEVICE)

    dataset = load_dataset("bavard/personachat_truecased")


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
    
    output_dir = './model_save/checkpoint_psize_64'

    # Create output directory if needed

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # print(model.get_prompt())

    # TOP_K = 100
    # TEMPERATURE = 0.6

    # with model.inference_session(max_length=512) as sess:
    #     while True:
    #         user_phrase = input()
    #         if len(user_phrase) == 0:
    #             break
    #         inputs = tokenizer([f"{user_phrase}\n-----\n"], return_tensors='pt')['input_ids']
    #         while True:
    #             outputs = model.generate(
    #                 inputs,
    #                 temperature=TEMPERATURE,
    #                 do_sample=True,
    #                 top_k=TOP_K,
    #                 max_new_tokens=1,
    #                 session=sess,
    #             )
    #             bloom_answer_token = tokenizer.decode(outputs[0, -1:])
    #             print(bloom_answer_token, end="", flush=True)
    #             if bloom_answer_token == "\n":
    #                 break
    #             inputs = None

if __name__ == "__main__":
    main()

