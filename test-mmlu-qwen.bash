#!/bin/bash

WINDOW=(2048 2048)
CASCADES=(4 1)
SINKS=(4 4)
BATCH_SIZE=5
HEAD_REDUCTION=max
CASCADE_FUNC="pow2"
CASCADE_STRIDE=128
GPU=5
# COMMENT="quarter-avg-len-budget"
COMMENT="half-ctx"
# MODEL=llama3.1-8b-instruct
MODEL=qwen2-7b-instruct
# --homogeneous_heads \

for i in "${!WINDOW[@]}";
do 
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job mmlu \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE \
        --cascade_stride $CASCADE_STRIDE
        sleep 1
done
