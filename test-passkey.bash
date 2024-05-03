#!/bin/bash

# passkey experiment
GPUS=(6 6)
WINDOW=(4096 4096)
CASCADES=(4 1)
SINKS=(4 4)
BATCH_SIZE=10
# MODEL=llama7b
# MODEL=llama3-8b-instruct
# MODEL=llama7b-chat
# MODEL=llama13b
# MODEL=qwen14b
#  MODEL=qwen7b
MODEL=qwen7b-chat

for i in "${!GPUS[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python timber/main/llama_eval.py \
        --model $MODEL \
        --job passkey \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE \
        --dev_run 
        
        sleep 1
done
