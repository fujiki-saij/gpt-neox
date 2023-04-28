#!/bin/bash
#SBATCH --job-name="jp-mc4"
#SBATCH --partition=cpu128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0 #0 means use all available memory (in MB)
#SBATCH --output=/fsx/home-mkshing/slurm_outs/%x_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/%x_%j.err
#SBATCH --comment stability

module load openmpi cuda/11.7
micromamba activate stable-neox-env-cpu

python /fsx/home-mkshing/code/jp-StableLM/gpt-neox/tools/preprocess_data.py \
    --input /fsx/jp-llm/polyglot-ja/2_quality_filter/v2/mc4 \
    --output-prefix /fsx/jp-llm/merged/mc4/mc4 \
    --vocab-file /fsx/jp-llm/novelai.model \
    --dataset-impl mmap \
    --tokenizer-type SPMTokenizer \
    --workers 128 \
    --append-eod