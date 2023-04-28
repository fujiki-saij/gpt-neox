# python ./deepy.py evaluate.py -d configs your_configs.yml --eval_tasks jcommonsenseqa jsquadqa jaquadqa

# What's changed
# - Apply JGLUE (with japanese prompt like "タイトル: {title} 背景: {context} 質問: {question} 回答: {answer}"). For more details, please see `lm_eval/tasks/jglue.py`. 
# - Enable tokenizer to set use_fast in gpt.py
# - remove invalid data inside JSQuAD (see `validation_docs` func in jglue.py)
MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,low_cpu_mem_usage=True"
# MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
# MODEL_ARGS="pretrained=rinna/japanese-gpt2-xsmall,use_fast=False"
TASK="jsquad,jaquad" # jsquad, jaquad, jcommonsenseqa
# source /fsx/home-mkshing/venv/nlp/bin/activate
python scripts/lm_harness_main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot 2 \
    --device "cuda" 