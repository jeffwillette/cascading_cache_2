#!/bin/bash

WINDOW=(1024)
CASCADES=(1)
SINKS=(4)
BATCH_SIZE=15
HEAD_REDUCTION=mean
CASCADE_FUNC="pow2"

for i in "${!WINDOW[@]}";
do 
        # PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python timber/main/llama_eval.py \
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 timber/main/llama_eval.py \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,3 --master_port 63290 timber/main/llama_eval.py \
        --model qwen7b \
        --job mmlu \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment rerun-since-deepspeed-bugfix \
        --batch_size $BATCH_SIZE
        sleep 1
done


# WINDOW=(1024)
# CASCADES=(2)
# SINKS=(4)
# BATCH_SIZE=20
# HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
# 
# # PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python timber/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 timber/main/llama_eval.py \
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
# # PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python timber/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 timber/main/llama_eval.py \
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
# # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. deepspeed --include localhost:0,1,2,3 --master_port 63390 timber/main/llama_eval.py \
# for i in "${!WINDOW[@]}";
# do 
#         PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 python timber/main/llama_eval.py \
#         --model llama7b \
#         --job mmlu \
#         --method vanilla \
#         --lora_r 0 \
#         --comment none \
#         --batch_size $BATCH_SIZE
#         sleep 1
# done
