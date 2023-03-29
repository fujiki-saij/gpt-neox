python lm-evaluation-harness/main.py \
    --model gpt2 \
    --model_args pretrained="EleutherAI/gpt-j-6B" \
    --tasks gsm8k \
    --device 0
    
    --model stable-lm \
    --model_args pretrained="CarperAI/test-7b-62k" \

    --tasks piqa,sciq,hellaswag,boolq,arc_easy,arc_challenge \