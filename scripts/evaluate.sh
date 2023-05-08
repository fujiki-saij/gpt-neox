#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="jp-eval"
#SBATCH --partition=g40
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=11G
#SBATCH --output=/fsx/home-mkshing/slurm_outs/%x_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/%x_%j.err


# MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,low_cpu_mem_usage=True"
MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
# MODEL_ARGS="pretrained=naclbit/gpt-j-japanese-6.8b,low_cpu_mem_usage=True" <- hasn't released the weight yet
TASK="jsquad,jaquad" # jsquad, jaquad, jcommonsenseqa
source /fsx/home-mkshing/venv/nlp/bin/activate
python scripts/lm_harness_main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot 2 \
    --device "cuda" \

# [Example] From gpt-neox's checkpoints
# python ./deepy.py evaluate.py \
#     -d configs /fsx/home-mkshing/code/jp-StableLM/gpt-neox/models/1b_test/global_step100/configs/stable-lm-jp-1b-experimental.yml \
#     --eval_tasks jcommonsenseqa,jsquadqa,jaquadqa \
#     --eval_num_fewshot 2