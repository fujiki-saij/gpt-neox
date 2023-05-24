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
    # expid = "exp001"  # TODO: change here
    expname = f"stablelm-jp-1b-ja50_rp50-460b-tuned_japanese_alpaca_data"
    # TODO: refactor as yaml config file
    # hyper parameters
    config = {
        "cache_dir": "/fsx/home-fujiki/.cache",  # TODO: change here
        "tokenizer_path": "/fsx/jp-llm/tokenizers/nai-hf-tokenizer",
        "model_path": "/fsx/jp-llm/hf_model/1b-ja50-rp50-460b",  # TODO: change here
        "save_dir": f"/fsx/jp-llm/instruction_tuning/outputs/{expname}",  # TODO: change here
        "train_args": {
            "output_dir": f"/fsx/jp-llm/instruction_tuning/outputs/{expname}",  # TODO: change here
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
            "logging_dir": f"/fsx/jp-llm/instruction_tuning/outputs/{expname}",  # TODO: change here
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
    main(config)