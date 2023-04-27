#!/bin/bash
#SBATCH --account="stability"
#SBATCH --job-name="jp-stable-lm"
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/fsx/home-mkshing/slurm_outs/neox_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/neox_%j.err

module load openmpi cuda/11.7
micromamba activate stable-neox-env

python /fsx/home-mkshing/code/jp-StableLM/gpt-neox/tools/preprocess_data.py \
    --input /fsx/jp-llm/polyglot-ja/2_quality_filter/v2/cc-100 \
    --output-prefix /fsx/jp-llm/merged/cc-100/cc-100 \
    --vocab-file /fsx/jp-llm/novelai.model \
    --dataset-impl mmap \
    --tokenizer-type SPMTokenizer \
    --append-eod