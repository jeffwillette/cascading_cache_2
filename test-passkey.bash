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

# passkey experiment
WINDOW=32768
# CASCADES=(8 1)
CASCADES=(1)
SINKS=64
BATCH_SIZE=2
HEAD_REDUCTION=max
CASCADE_STRIDE=4096
CASCADE_FUNC="pow2"
GPU=$gpu
MODEL=$model
METHOD=$method

for i in "${!CASCADES[@]}";
do 
    # PYTHONPATH=. deepspeed --include localhost:3,4,5,6 --master_port 63280 cascade/main/llama_eval.py \
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
        --cascades ${CASCADES[$i]} \
        --batch_size $BATCH_SIZE
        sleep 1
done
