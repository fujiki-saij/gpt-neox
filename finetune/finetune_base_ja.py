#!/usr/bin/env python
# coding: utf-8
import os

import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from datasets import load_dataset
import transformers as T
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from rm_datasets import SFTDataset, MaskedSFTDataset


# TODO: refactor
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_text_len=4096, cache_name="train"):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.prompts = []
        if hasattr(tokenizer, "eos_token_id"):
            EOS_ID = tokenizer.eos_token_id
            EOS_TOKEN = tokenizer.eos_token
        else:
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
            EOS_TOKEN = '<|endoftext|>'

        max_length = max_text_len
        print("Max length: {}".format(max_length))

        # TODO: clean up comment outed lines
        # # Data expected in prompt response pairs
        #     if os.path.exists(cache_name + "inputids.pt"):
        #         print("using cached dataset")
        #         self.input_ids = torch.load(cache_name+"inputids.pt")
        #         self.attn_masks = torch.load(cache_name+"attnmask.pt")
        #         self.labels = torch.load(cache_name+"inputids.pt")
        #         return
        for ele in tqdm(data):
            prompt = ele["text"]
            prompt_encoding_len = len(tokenizer(prompt)["input_ids"])
            encodings_dict = tokenizer(
                prompt, truncation=True, max_length=max_length, padding="max_length")
            input_id = torch.tensor(encodings_dict['input_ids'])
            attn_mask = torch.tensor(encodings_dict['attention_mask'])
            self.input_ids.append(input_id)
            self.attn_masks.append(attn_mask)
            self.labels.append(input_id)
            # torch.save(self.input_ids, cache_name+"inputids.pt")
            # torch.save(self.attn_masks, cache_name+"attnmask.pt")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


if __name__ == "__main__":
    expid = "exp001"  # TODO: change here
    expname = f"{expid}_stablelm-jp-0.0.0-a.1_tuned_japanese_alpaca_data"
    # TODO: refactor as yaml config file
    # hyper parameters
    config = {
        "cache_dir": "/path/to/cache/directory",  # TODO: change here
        "tokenizer_path": "/fsx/jp-llm/tokenizers/nai-hf-tokenizer",
        "model_path": "/fsx/jp-llm/hf_model/test",
        "save_dir": f"/path/to/output/directory/{expname}",  # TODO: change here
        "train_args": {
            "output_dir": f"/path/to/output/directory/{expname}",  # TODO: change here
            "num_train_epochs": 2,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "fp16": True,
            "learning_rate": 1.0e-7,
            "lr_scheduler_type": "constant",
            # "adam_beta2": 0.99,
            # "weight_decay": 0.01,
            # "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            # "warmup_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "logging_dir": f"/path/to/output/log/directory/{expname}",  # TODO: change here
            "logging_steps": 100,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 2,
        },
        "data_path": "fujiki/japanese_alpaca_data",
        "train_size": 0.98,
        "trainer": "text",
        "max_text_len": 1024,
    }

    # data
    raw_data = load_dataset(config["data_path"], cache_dir=config["cache_dir"])['train']
    # tokenizer
    if "rinna" in config["tokenizer_path"]:
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"], use_fast=False, cache_dir=config.get("cache_dir", None),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"], cache_dir=config.get("cache_dir", None),
        )
    tokenizer.pad_token = tokenizer.eos_token
    # model
    model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to cuda")
    training_args = TrainingArguments(**config["train_args"])

    # prompt templates
    INPUT_PROMPT = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
{instruction}

### 入力: 
{input}

### 応答: 
{response}"""

    NO_INPUT_PROMPT = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
{instruction}

### 応答: 
{response}"""

    def preprocess_function(example):
        # remove parenthesis that might be introduced by some NMTs
        new_example = {}
        for k, v in example.items():
            if example[k].startswith("「") and example[k].endswith("」"):
                new_example[k] = example[k].strip("「").strip("」")
            else:
                new_example[k] = example[k]
        return new_example

    processed_data = raw_data.map(
        preprocess_function,
        batched=False,
    )
    assert len(raw_data) == len(processed_data)

    def make_text_field(example):
        # put alapaca-like formatted data point into the prompt template
        output_field_name = 'output'
        if example['input']:
            example['text'] = INPUT_PROMPT.format(
                instruction=example['instruction'], input=example['input'], response=example[output_field_name])
        else:
            example['text'] = NO_INPUT_PROMPT.format(
                instruction=example['instruction'], input=example['input'], response=example[output_field_name])
        return example

    data = processed_data.map(
        make_text_field,
        batched=False,
        remove_columns=["instruction", "input", "output"]
    )

    if config["trainer"] == "unmasked":
        dataset = SFTDataset(data, tokenizer)
    elif config["trainer"] == "masked":
        dataset = MaskedSFTDataset(data, tokenizer)
    elif config['trainer'] == 'text':
        cache_name = config["data_path"].replace("/", "-")
        dataset = TextDataset(data, tokenizer, config['max_text_len'], cache_name=cache_name)
        print("Load TextDataset")

    train_size = int(config['train_size'] * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=lambda data: {
            'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[2] for f in data])
        }
    )
    print("Start training...")
    trainer.train()