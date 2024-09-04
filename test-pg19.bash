#!/bin/bash

WINDOW=16384
CASCADES=4
SINKS=4
BATCH_SIZE=1
HEAD_REDUCTION=(mean)
CASCADE_FUNC="pow2"
GPUS=(4)
MODEL=llama3.1-8b-instruct
# MODEL=qwen2-7b
# MODEL=llama3.1-70b
# MODEL=llama3.1-70b-instruct-gptq-int4
METHOD=sink
# COMMENT="different-cache-sizes-65k-4-8k-28"
# COMMENT="different-cache-sizes-65k-4-16k-28"
# COMMENT="different-cache-sizes-131k-2-8k-30"
# COMMENT="window-quarter-book-score-correct-32768all"
COMMENT="16384all"
# COMMENT="window-half-book"
# COMMENT="plain"
CASCADE_STRIDE=1024

# MAIN PG19 experiment code
for i in "${!HEAD_REDUCTION[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
        --model $MODEL \
        --job ppl-pg19 \
        --method $METHOD \
        --window $WINDOW \
        --sinks $SINKS \
        --cascades $CASCADES \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction ${HEAD_REDUCTION[$i]} \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE \
        --homogeneous_heads
        sleep 1
done

# for i in "${!WINDOW[@]}";
# do 
#     PYTHONPATH=. deepspeed --include localhost:2,3,4,5 --master_port 63290 cascade/main/llama_eval.py \
#         --model $MODEL \
#         --job ppl-pg19 \
#         --method $METHOD \
#         --window ${WINDOW[$i]} \
#         --sinks ${SINKS[$i]} \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --cascade_stride $CASCADE_STRIDE \
#         --head_reduction $HEAD_REDUCTION \
#         --comment $COMMENT \
#         --batch_size $BATCH_SIZE
#         # --homogeneous_heads
#         sleep 1
# done

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
