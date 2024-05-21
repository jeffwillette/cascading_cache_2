#!/bin/bash

# passkey experiment
GPUS=(0)
WINDOW=(512)
CASCADES=(8)
SINKS=(32)
BATCH_SIZE=50

# GPUS=(2)
# WINDOW=(512)
# CASCADES=(1)
# SINKS=(32)
# BATCH_SIZE=50

# GPUS=(7)
# WINDOW=(1024)
# CASCADES=(1)
# SINKS=(4)
# BATCH_SIZE=50
HEAD_REDUCTION=mean
# MODEL=llama7b
MODEL=llama3-8b-instruct
# MODEL=llama7b-chat
# MODEL=llama13b
# MODEL=qwen14b
# MODEL=qwen7b
# MODEL=qwen7b-chat

for i in "${!GPUS[@]}";
do 
    # PYTHONPATH=. deepspeed --include localhost:2,0 --master_port 63280 cascade/main/llama_eval.py \
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
        --model $MODEL \
        --job passkey \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --head_reduction $HEAD_REDUCTION \
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE
        
        sleep 1
done
