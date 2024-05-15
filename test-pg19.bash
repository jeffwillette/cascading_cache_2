#!/bin/bash

WINDOW=(2048)
CASCADES=(1)
SINKS=(4)
BATCH_SIZE=5
HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
CASCADE_FUNC="pow2"
GPUS=(0)

for i in "${!WINDOW[@]}";
do 
    # PYTHONPATH=. deepspeed --include localhost:0,3 --master_port 63290 timber/main/llama_eval.py \
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python timber/main/llama_eval.py \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,3 python timber/main/llama_eval.py \
        --model llama3-70b-instruct \
        --job ppl-pg19 \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment llama3-70b \
        --batch_size $BATCH_SIZE
        sleep 1
done
