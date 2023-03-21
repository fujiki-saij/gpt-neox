python lm-evaluation-harness/main.py \
    --model stable-lm \
    --model_args pretrained="" \
    --tasks piqa,sciq,hellaswag,boolq,arc_easy,arc_challenge \
    --device 0