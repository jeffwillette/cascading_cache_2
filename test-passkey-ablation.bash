#!/bin/bash

GPUS=4
WINDOW=4096
# CASCADES=(1 2)
CASCADES=(8 4)
# CASCADES=(32 16)
SINKS=64
BATCH_SIZE=10
HEAD_REDUCTION=max
MODEL=llama3.1-8b-instruct
CASCADE_FUNC=pow2
CASCADE_STRIDE=1024
COMMENT=262K

for i in "${!CASCADES[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPUS python cascade/main/llama_eval.py \
        --model $MODEL \
        --job passkey \
        --method sink \
        --lora_r 0 \
        --window $WINDOW \
        --sinks $SINKS \
        --head_reduction $HEAD_REDUCTION \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --comment $COMMENT \
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE
        
        sleep 1
done
