#!/bin/bash

WINDOW=(4096)
CASCADES=(4)
SINKS=(4)
BATCH_SIZE=6
HEAD_REDUCTION=none
CASCADE_FUNC="pow2"

for i in "${!WINDOW[@]}";
do 
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python timber/main/llama_eval.py \
    PYTHONPATH=. deepspeed --include localhost:4,5,6,7 --master_port 63290 timber/main/llama_eval.py \
        --model llama7b \
        --job ppl-pg19 \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment llama7b-no-head-reduction \
        --batch_size $BATCH_SIZE
        sleep 1
done

# GPUS=(5)
# WINDOW=(128)
# CASCADES=(4)
# SINKS=(4)
# 
# for i in "${!GPUS[@]}";
# do 
#     PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python timber/main/llama_eval.py \
#         --model llama32k \
#         --job mmlu \
#         --method sink \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]}
#         --batch_size $BATCH_SIZE \
#         
#         sleep 1
# done

# llama 7b experiment
# GPUS=(2)
# WINDOW=(16)
# CASCADES=(4)
# SINKS=(4)
# 
# for i in "${!GPUS[@]}";
# do 
#     PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python timber/main/llama_eval.py \
#         --model llama32k \
#         --job ppl \
#         --method sink \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --batch_size $BATCH_SIZE \
#         
#         sleep 1
# done
