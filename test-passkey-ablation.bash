#!/bin/bash

# GPUS=0
# WINDOW=4096
# CASCADES=(1 2 4 8 8 8 8 8 8 8)
# SINKS=64
# BATCH_SIZE=1
# HEAD_REDUCTION=(mean mean mean mean mean mean mean mean median max)
# MODEL=llama3.1-8b-instruct
# CASCADE_FUNCS=("pow2" "pow2" "pow2" "pow2" "pow2-1" "pow2-1-4" "pow2-2-4" "pow2-3-4" "pow2" "pow2")
#
GPUS=0
WINDOW=4096
CASCADES=(8 8 8 8 8)
SINKS=64
BATCH_SIZE=1
HEAD_REDUCTION=(mean mean mean median max)
MODEL=llama3.1-8b-instruct
CASCADE_FUNCS=("pow2-1-4" "pow2-2-4" "pow2-3-4" "pow2" "pow2")

for i in "${!CASCADES[@]}";
do 
    # PYTHONPATH=. deepspeed --include localhost:3,4,5,6 --master_port 63280 cascade/main/llama_eval.py \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPUS python cascade/main/llama_eval.py \
        --model $MODEL \
        --job passkey \
        --method sink \
        --lora_r 0 \
        --window $WINDOW \
        --sinks $SINKS \
        --head_reduction ${HEAD_REDUCTION[$i]} \
        --cascade_func ${CASCADE_FUNCS[$i]} \
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE
        
        sleep 1
done
