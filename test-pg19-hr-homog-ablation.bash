#!/bin/bash

model=invalid-model-name
method=sink
while getopts m:d:g: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) method=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

GPU=$gpu
MODEL=$model
METHOD=$method
HOMOGENEOUS_HEADS=("" "" "" "--homogeneous_heads" "--homogeneous_heads" "--homogeneous_heads")
HEAD_REDUCTION=(mean median max mean median max)

if [ "$METHOD" = "vanilla" ]; then
    HOMOGENEOUS_HEADS=""
    HEAD_REDUCTION=(mean)
fi

WINDOW=16384
CASCADES=4
SINKS=4
BATCH_SIZE=1
CASCADE_FUNC="pow2"
COMMENT="none"
CASCADE_STRIDE=1024

# MAIN PG19 experiment code
for i in "${!HEAD_REDUCTION[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
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
        ${HOMOGENEOUS_HEADS[$i]} \
        --batch_size $BATCH_SIZE
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
