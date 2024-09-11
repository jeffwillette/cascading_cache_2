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

MODEL=$model
METHOD=$method
WINDOW=(2048 2048)
CASCADES=(4 1)
SINKS=(4 4)
COMMENT=("2048-all" "2048-all")
BATCH_SIZE=20

if [ "$METHOD" = "vanilla" ]; then
    WINDOW=(2048 2048)
    CASCADES=(1 1)
    SINKS=(4 4)
    BATCH_SIZE=1
    COMMENT=("vanilla-unconstrained" "vanilla-truncate")
fi

HEAD_REDUCTION=max
CASCADE_FUNC="pow2"
CASCADE_STRIDE=512
GPU=$gpu
# --homogeneous_heads \

for i in "${!WINDOW[@]}";
do 
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model $MODEL \
        --job mmlu \
        --method $METHOD \
        --lora_r 0 \
        --window ${WINDOW[$i]} \
        --sinks ${SINKS[$i]} \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --head_reduction $HEAD_REDUCTION \
        --comment ${COMMENT[$i]} \
        --batch_size $BATCH_SIZE \
        --cascade_stride $CASCADE_STRIDE
        sleep 1
done
