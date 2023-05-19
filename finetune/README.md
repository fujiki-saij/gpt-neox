# fine-tuning codebase for Japanese LLMs
## How to use
- By running `finetune_base_ja.py`, you can finetune some LLMs on Japanese translation version of Alpaca dataset.
  - Please change some configuration in the file according to your need.
- `finetune_base_ja.py` is based on [dmahan93](https://github.com/dmahan93)'s codes:
    - [finetune_base.py](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/reward-modeling/finetune_base.py)
    - [rm_dataset.py](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/reward-modeling/rm_datasets.py)
    - [stable6B.yaml](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/configs/pythia/base_configs/stable6B.yaml)
