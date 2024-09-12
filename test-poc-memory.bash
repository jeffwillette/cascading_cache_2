#!/bin/bash

# memory experiment
GPUS=(1 1)
WINDOW=(1024 1024)
CASCADES=(4 1)
SINKS=(64 64)
BATCH_SIZE=1
# MODEL=llama7b
# MODEL=llama13b
MODEL=qwen14b
# MODEL=qwen7b

for i in "${!GPUS[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
        --model $MODEL \
        --job ppl-memory \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE
        
        sleep 1
done
