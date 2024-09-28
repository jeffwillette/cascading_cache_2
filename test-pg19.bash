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
HEAD_REDUCTION=max

WINDOW=(32768 65536 32768 65536)
CASCADES=(4 4 1 1)

SINKS=64
BATCH_SIZE=1
CASCADE_FUNC="pow2"
COMMENT="none"
CASCADE_STRIDE=1024

# TEMPORARY ADDITION TO RUN BOTH MODELS FOR 16K
# WINDOW=(16384 16384)
# CASCADES=(1 1)
# MODEL=(llama3.1-8b qwen2-7b)

# TEMPORARY ADDITION FOR RUNNING VANILLA STRIDED
# CASCADE_STRIDE=(32768 16384)
# WINDOW=(32768 16384)
# CASCADES=(1 1)

# TEMPORARY ADDITION FOR RUNNING BIGBIRD
# CASCADE_STRIDE=(32768 16384)
# WINDOW=(32768 16384)
# CASCADES=(1 1)

# TEMPORARY ADDITION TO RUN H2O
# WINDOW=(16384 16384)
# WINDOW=(65536)
# CASCADES=(1 1 1 1 1)
# CASCADE_STRIDE=(128 256 512 1024 1)
# COMMENT=h2o-linear-stride-vs-single-step-ablation
# COMMENT=stride-vs-single-step-ablation
# WINDOW=(2048 2048 2048 2048 2048)

# for patching up single step experiment
CASCADES=(1)
CASCADE_STRIDE=(1)
COMMENT=h2o-linear-stride-vs-single-step-ablation
WINDOW=(2048)


# MAIN PG19 experiment code
for i in "${!WINDOW[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job ppl-pg19 \
        --method $METHOD \
        --window ${WINDOW[$i]} \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride ${CASCADE_STRIDE[$i]} \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done


# For llama/qwen 16K
# GPU=3
# MODEL=("llama3.1-8b" "qwen2-7b")
# METHOD=sink
# HEAD_REDUCTION=max
# 
# WINDOW=(16384 16384)
# CASCADES=(1 1)
# 
# SINKS=64
# BATCH_SIZE=1
# CASCADE_FUNC="pow2"
# COMMENT="none"
# CASCADE_STRIDE=1024
# for i in "${!WINDOW[@]}";
# do 
#     PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
#         --model ${MODEL[$i]} \
#         --job ppl-pg19 \
#         --method $METHOD \
#         --window ${WINDOW[$i]} \
#         --sinks $SINKS \
#         --cascades ${CASCADES[$i]} \
#         --cascade_func $CASCADE_FUNC \
#         --cascade_stride $CASCADE_STRIDE \
#         --head_reduction $HEAD_REDUCTION \
#         --comment $COMMENT \
#         --batch_size $BATCH_SIZE
#         sleep 1
# done
