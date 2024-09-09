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

WINDOW=2048
CASCADES=(4 1)
METHOD=$method

if [ "$METHOD" = "vanilla" ]; then
    CASCADES=(1)
fi

SINKS=4
BATCH_SIZE=1
HEAD_REDUCTION=max
CASCADE_FUNC="pow2"
GPU=$gpu
MODEL=$model
COMMENT="vanilla-truncate-right-half-ctx"
CASCADE_STRIDE=512

for i in "${!CASCADES[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job booksum \
        --method $METHOD \
        --window $WINDOW \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
