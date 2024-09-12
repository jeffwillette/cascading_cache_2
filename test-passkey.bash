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

# passkey experiment
WINDOW=32768
SINKS=64
BATCH_SIZE=2
HEAD_REDUCTION=max
CASCADE_STRIDE=4096
CASCADE_FUNC="pow2"
CASCADES=$cascades
GPU=$gpu
MODEL=$model
METHOD=$method

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
    --model $MODEL \
    --job passkey \
    --method $METHOD \
    --lora_r 0 \
    --window $WINDOW \
    --sinks $SINKS \
    --head_reduction $HEAD_REDUCTION \
    --cascade_func $CASCADE_FUNC \
    --cascade_stride $CASCADE_STRIDE \
    --cascades $CASCADES \
    --batch_size $BATCH_SIZE
