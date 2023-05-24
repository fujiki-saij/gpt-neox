#!/usr/bin/env python
# coding: utf-8
import os

import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from datasets import load_dataset
import transformers as T
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from rm_datasets import SFTDataset, MaskedSFTDataset, TextDataset
from templates import INPUT_PROMPT, NO_INPUT_PROMPT
from utils import load_yaml


def main(config):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/sample.yaml")
    args, _ = parser.parse_known_args()
    config = load_yaml(args.config_path)
    main(config)