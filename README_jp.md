# JP StableLM
## Installation
Follow [GPT-NeoX Setup Guide for Stable-LM](https://docs.google.com/document/d/1vV4fPgv5OkBtoTqAR86f3i4_xWIbnJN3td38g8m6ieY/edit)

Make sure to use `requirements/requirements.txt` in `jp-stable` branch to use the custom `lm_eval`. 

## Evaluation
Supported JGLUE/JaQuAD. 
- In the [Ricoh's paper](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf), `jcommonsenseqa` is 3-shot setting (`--num_fewshot 3`) and `jsquad` is 2-shot setting (`--num_fewshot 2`).  
### Using HF checkpoints
```
MODEL_ARGS="pretrained=abeja/gpt-neox-japanese-2.7b,low_cpu_mem_usage=True"
TASK="jsquad,jaquad" # jsquad, jaquad, jcommonsenseqa
python scripts/lm_harness_main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot 2 \
    --device "cuda" 
```

### Using gpt-neox checkpoints
```bash
MODEL_CONFIG=/fsx/home-mkshing/code/jp-StableLM/gpt-neox/models/1b_test/global_step100/configs/stable-lm-jp-1b-experimental.yml
python ./deepy.py evaluate.py \
    -d configs $MODEL_CONFIG \
    --eval_tasks jcommonsenseqa \
    --eval_num_fewshot 3
```