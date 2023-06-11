#!/usr/bin/env python
# coding: utf-8
import os

import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from datasets import concatenate_datasets, DatasetDict, load_dataset
import transformers as T
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from rm_datasets import SFTDataset, MaskedSFTDataset, TextDataset
from templates import INPUT_PROMPT, NO_INPUT_PROMPT
from utils import load_yaml


def main(config):
    # data
    if isinstance(config["data_path"], list):
        # load multiple datasets
        raw_datasets = []
        for data_path in config['data_path']:
            ds = load_dataset(data_path, cache_dir=config['cache_dir'])['train']
            print(f"Loaded `{data_path}`. The dataset size is {len(ds)}")
            ds = ds.train_test_split(test_size=(1 - config['train_size']), seed=config.get("seed", 42))
            raw_datasets.append(ds)

        # concatenate the multiple datasets
        raw_dataset = DatasetDict()
        for split in ['train', 'test']:
            raw_dataset[split] = concatenate_datasets([dataset[split] for dataset in raw_datasets])
    else:
        raw_dataset = load_dataset(config["data_path"], cache_dir=config["cache_dir"])['train']
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - config['train_size']), seed=config.get("seed", 42))
    print(f"Train set: {len(raw_dataset['train'])}")
    print(f"Test set: {len(raw_dataset['test'])}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["tokenizer_name_or_path"], use_fast=config["tokenizer"].get("use_fast", False),
        cache_dir=config.get("cache_dir", None),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to cuda")
    training_args = TrainingArguments(**config["train_args"])

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

    column_names = list(raw_dataset["train"].features)
    text_dataset = raw_dataset.map(
        make_text_field,
        batched=False,  # TODO: enable batched=True
        remove_columns=column_names,
    )
    print(f"Finish making text field")

    if config["trainer"] == "unmasked":
        text_dataset['train'] = SFTDataset(text_dataset['train'], tokenizer)
        text_dataset['test'] = SFTDataset(text_dataset['test'], tokenizer)
    elif config["trainer"] == "masked":
        text_dataset['train'] = MaskedSFTDataset(text_dataset['train'], tokenizer)
        text_dataset['test'] = MaskedSFTDataset(text_dataset['test'], tokenizer)
    elif config['trainer'] == 'text':
        cache_name = os.path.basename(training_args.output_dir)
        text_dataset['train'] = TextDataset(text_dataset['train'], tokenizer, config['max_text_len'], cache_name=cache_name)
        text_dataset['test'] = TextDataset(text_dataset['test'], tokenizer, config['max_text_len'], cache_name=cache_name)
        print("Loaded TextDataset")

    model.config.use_cache = False
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=text_dataset['train'],
        eval_dataset=text_dataset['test'],
        data_collator=lambda data: {
            'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[2] for f in data])
        }
    )
    print("Start training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/sample.yaml")
    args, _ = parser.parse_known_args()
    config = load_yaml(args.config_path)
    main(config)