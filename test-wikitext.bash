#!/bin/bash

model=invalid-model-name
method=sink
cascades=1
while getopts m:d:g:c: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) method=${OPTARG};;
        g) gpu=${OPTARG};;
        c) cascades=${OPTARG};;
    esac
done

GPU=$gpu
MODEL=$model
METHOD=$method
HEAD_REDUCTION=max

WINDOW=(128 256 512 1024 2048)
CASCADES=$cascades

SINKS=64
BATCH_SIZE=1
CASCADE_FUNC="pow2"
COMMENT="none"
CASCADE_STRIDE=(128 256 512 1024 2048)

# MAIN PG19 experiment code
for i in "${!WINDOW[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job ppl-wikitext \
        --method $METHOD \
        --window ${WINDOW[$i]} \
        --sinks $SINKS \
        --cascades $CASCADES \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride ${CASCADE_STRIDE[$i]} \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
