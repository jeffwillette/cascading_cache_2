#!/bin/bash

WINDOW=(16384)
CASCADES=(1)
SINKS=(64)
BATCH_SIZE=1
HEAD_REDUCTION=mean
CASCADE_FUNC="pow2"
GPUS=(0)
MODEL=llama3.1-8b
METHOD=vanilla
COMMENT="llama3.1"
CASCADE_STRIDE=16384

# MAIN PG19 experiment code
for i in "${!WINDOW[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
        --model $MODEL \
        --job ppl \
        --method $METHOD \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
