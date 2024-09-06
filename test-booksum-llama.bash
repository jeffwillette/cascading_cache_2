#!/bin/bash

WINDOW=2048
CASCADES=(4 1)
SINKS=4
BATCH_SIZE=1
HEAD_REDUCTION=max
CASCADE_FUNC="pow2"
GPUS=(5 5)
MODEL=llama3.1-8b-instruct
METHOD=sink
COMMENT="half-ctx"
CASCADE_STRIDE=512

for i in "${!GPUS[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
        --model $MODEL \
        --job booksum \
        --method $METHOD \
        --window $WINDOW \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
