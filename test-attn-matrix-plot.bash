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
WINDOW=(2048 2048)
CASCADES=(4 1)

SINKS=64
BATCH_SIZE=1
CASCADE_FUNC="pow2"
COMMENT="none"
CASCADE_STRIDE=256

# MAIN PG19 experiment code
# --do_og_pos \
for i in "${!WINDOW[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job attn_matrix_plot \
        --method $METHOD \
        --window ${WINDOW[$i]} \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction $HEAD_REDUCTION \
        --do_og_pos \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
