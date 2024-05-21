#!/bin/bash

WINDOW=(2048)
CASCADES=(4)
SINKS=(4)
BATCH_SIZE=25
HEAD_REDUCTION=independent
# CASCADE_FUNC="pow2"
CASCADE_FUNC="pow2"
GPUS=(0)

# MAIN PG19 experiment code
for i in "${!WINDOW[@]}";
do 
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,3 python cascade/main/llama_eval.py \
    PYTHONPATH=. deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 63290 cascade/main/llama_eval.py \
        --model llama7b \
        --job ppl-pg19 \
        --method sink \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment llama7b-independent-heads \
        --batch_size $BATCH_SIZE
        sleep 1
done

# WINDOW=(2048)
# CASCADES=(1)
# SINKS=(4)
# BATCH_SIZE=1
# HEAD_REDUCTION=mean
# CASCADE_FUNC="pow2"
# GPUS=(0)
# 
# # HYPER ATTENTION BASELINE ===================================================
# for i in "${!WINDOW[@]}";
# do 
#     # PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
#     # PYTHONPATH=. deepspeed --include localhost:0,1,2 --master_port 63290 cascade/main/llama_eval.py \
#     # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python cascade/main/llama_eval.py \
#     #     --model qwen7b \
#     #     --job ppl-pg19 \
#     #     --method hyper \
#     #     --lora_r 0 \
#     #     --window ${WINDOW[$i]} \
#     #     --sinks ${SINKS[$i]} \
#     #     --cascades ${CASCADES[$i]} \
#     #     --cascade_func $CASCADE_FUNC \
#     #     --head_reduction $HEAD_REDUCTION \
#     #     --comment qwen7b-hyper-attention \
#     #     --batch_size $BATCH_SIZE \
#     #     --dev_run 
#     #     sleep 1
# 
#     # PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python cascade/main/llama_eval.py \
#     PYTHONPATH=. deepspeed --include localhost:4,5 --master_port 63290 cascade/main/llama_eval.py \
#         --model qwen14b \
#         --job ppl-pg19 \
#         --method hyper \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --head_reduction $HEAD_REDUCTION \
#         --comment qwen14b-quadratic \
#         --batch_size $BATCH_SIZE \
#         --dev_run 
#         sleep 1
# done


# ATTN MATRIC PLOTTING ========================================================
# WINDOW=(2048)
# CASCADES=(1)
# SINKS=(4)
# BATCH_SIZE=1
# HEAD_REDUCTION=mean
# # CASCADE_FUNC="pow2"
# CASCADE_FUNC="pow2"
# GPUS=(1)
# 
# for i in "${!WINDOW[@]}";
# do 
#     # PYTHONPATH=. deepspeed --include localhost:0,3 --master_port 63290 cascade/main/llama_eval.py \
#     # PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
#     PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python cascade/main/llama_eval.py \
#         --model llama7b \
#         --job ppl-pg19 \
#         --method sink \
#         --lora_r 0 \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --head_reduction $HEAD_REDUCTION \
#         --comment llama7b-attention-matrix-plot \
#         --batch_size $BATCH_SIZE \
#         --dev_run
#         sleep 1
# done
