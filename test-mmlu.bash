#!/bin/bash

WINDOW=(2048)
CASCADES=(4)
SINKS=(4)
BATCH_SIZE=10
HEAD_REDUCTION=mean
CASCADE_FUNC="pow2"
CASCADE_STRIDE=512
GPU=4
# COMMENT="quarter-avg-len-budget"
COMMENT="quarter-avg-len-budget"
# MODEL=llama3.1-8b-instruct
MODEL=qwen2-7b-instruct

for i in "${!WINDOW[@]}";
do 
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 cascade/main/llama_eval.py \
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,3 --master_port 63290 cascade/main/llama_eval.py \
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
        --homogeneous_heads \
        --cascade_stride $CASCADE_STRIDE
        sleep 1
done


# WINDOW=(1024)
# CASCADES=(2)
# SINKS=(4)
# BATCH_SIZE=20
# HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
# 
# # PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python cascade/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 cascade/main/llama_eval.py \
#         --model llama7b \
#         --job mmlu \
#         --method sink \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --head_reduction $HEAD_REDUCTION \
#         --comment none \
#         --batch_size $BATCH_SIZE
#         sleep 1
# done

# WINDOW=(1024)
# CASCADES=(1)
# SINKS=(4)
# BATCH_SIZE=5
# HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
# 
# # PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python cascade/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 cascade/main/llama_eval.py \
#         --model qwen7b \
#         --job mmlu \
#         --method sink \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --head_reduction $HEAD_REDUCTION \
#         --comment none \
#         --batch_size $BATCH_SIZE
#         sleep 1
# done

# BATCH_SIZE=1
# HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
# 
# # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3 --master_port 63390 cascade/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 python cascade/main/llama_eval.py \
#         --model llama7b \
#         --job mmlu \
#         --method vanilla \
#         --lora_r 0 \
#         --comment none \
#         --batch_size $BATCH_SIZE
#         sleep 1
# done
