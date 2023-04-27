"""
Compute number of tokens in wikipedia by a tokenizer
1. Firstly, prepare wikipedia by following https://github.com/rinnakk/japanese-pretrained-models#data-construction-and-model-training.

"""
import os
import json
from tqdm import tqdm
import glob
from transformers import AutoTokenizer, T5Tokenizer


NOVELAI = "/fsx/home-mkshing/models/novelai-tokenizer"


def convert_novelai_to_hf():
    # https://github.com/NovelAI/novelai-tokenizer
    novel_ai_sp = "novelai.model"
    tokenizer = T5Tokenizer(
        novel_ai_sp,
        unk_token="<|unknown|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        extra_ids=0,
    )
    tokenizer.save_pretrained("novelai-tokenizer")
    # make sure to load by T5Tokenizer
    # AutoTokenizer calls T5TokenizerFast which is based on unigram. https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5TokenizerFast
    # NovelAI Tokenizer is based on BPE. 
    tokenizer = T5Tokenizer.from_pretrained("noveai-tokenizer")


def main():
    # tokenizer_list = ["abeja/gpt-neox-japanese-2.7b", "rinna/japanese-gpt2-small", "rinna/japanese-gpt-1b", "StabilityAI/stablelm-base-alpha-3b", "naclbit/gpt-j-japanese-6.8b", "cl-tohoku/bert-base-japanese-v2", "novelai"]
    tokenizer_list = ["novelai"]
    wiki_files = glob.glob("/fsx/home-mkshing/data/jp_wiki/doc_data/*.txt")
    MAX_SAVE_LINES_FOR_UNK = 5000
    OUTPUT_DIR = "logs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_id in tqdm(tokenizer_list):
        print(f"model_id: {model_id}")
        outfile = os.path.join(OUTPUT_DIR, model_id.replace("/", "-")+".json")
        if os.path.exists(outfile):
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_id) if "novelai" not in model_id else T5Tokenizer.from_pretrained(NOVELAI)
        summary = {"class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer), "n_tokens": 0, "n_unk": 0, "unk_lines": []}
        for file in tqdm(wiki_files):
            with open(file,  "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    input_ids = tokenizer.encode(line)
                    summary['n_tokens'] += len(input_ids)
                    unk_count = input_ids.count(tokenizer.unk_token_id)
                    if unk_count > 0:
                        summary['n_unk'] += unk_count
                        if len(summary['unk_lines']) <= MAX_SAVE_LINES_FOR_UNK:
                            summary['unk_lines'].append(line)
        with open(outfile, "w", encoding="utf-8-sig") as fw:
            fw.write(json.dumps(summary, indent=4))
        del tokenizer


if __name__ == "__main__":
    main()
