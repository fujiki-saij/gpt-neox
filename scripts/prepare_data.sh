#!/bin/bash
#SBATCH --job-name="2023"
#SBATCH --partition=cpu128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=128
####SBATCH --mem=0 #0 means use all available memory (in MB)
#SBATCH --output=/fsx/home-mkshing/slurm_outs/%x_%j.out
#SBATCH --error=/fsx/home-mkshing/slurm_outs/%x_%j.err
#SBATCH --comment stablegpt
#SBATCH --exclusive

source /fsx/home-mkshing/.bashrc
micromamba activate stable-neox-env-cpu

# INPUT_PATH=/fsx/jp-llm/polyglot-ja/2_quality_filter/v2/mc4
# OUTPUT_PATH=/fsx/jp-llm/merged/mc4/mc4
# RedPajama
# INPUT_PATH="/fsx/redpajama/common-crawl-by-year/2019"
# OUTPUT_PATH=/fsx/jp-llm/merged/redpajama/common-crawl-by-year/2019/commoncrawl

echo "running code"
DATA=/fsx/redpajama
OUTDATA=/fsx/jp-llm/merged/redpajama
PROJDIR=common-crawl-by-year/2023
TMPDIR=/scratch

cd /fsx/home-mkshing/code/jp-StableLM/gpt-neox
FILES=$(ls $DATA/$PROJDIR/*.zst* | tr '\n' ' ' | sed 's/ /,/g')
FILES=${FILES:0:-1}

if [ ! -d "$OUTDATA/$PROJDIR" ]; then
  echo Creating directory: $OUTDATA/$PROJDIR
  mkdir -p "$OUTDATA/$PROJDIR"
fi

python tools/preprocess_data.py \
    --input $FILES \
    --output-prefix "$OUTDATA/$PROJDIR/tokenized" \
    --vocab-file /fsx/jp-llm/novelai.model \
    --dataset-impl mmap \
    --tokenizer-type SPMTokenizer \
    --workers 128 \
    --append-eod \
    --n_chunks 4

echo "finished running"

#2019 -> tail -f /fsx/home-mkshing/slurm_outs/2019_21224.err
#2020 -> tail -f /fsx/home-mkshing/slurm_outs/2020_21227.err
#2021 -> tail -f /fsx/home-mkshing/slurm_outs/2021_21228.err
#2022 -> tail -f /fsx/home-mkshing/slurm_outs/2022_21229.err
#2023 -> tail -f /fsx/home-mkshing/slurm_outs/2023_21231.err