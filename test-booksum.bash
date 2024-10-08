#!/bin/bash

model=invalid-model-name
method=sink
cascade_stride=512
while getopts m:d:g:c: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) method=${OPTARG};;
        g) gpu=${OPTARG};;
        c) cascade_stride=${OPTARG};;
    esac
done

WINDOW=2048
CASCADES=(4 1)

if [[ "$method" = "vanilla" || "$method" = "snapkv" || "$method" = "bigbird" || "$method" = "h2o" ]]; then
    CASCADES=(1)
fi

SINKS=64
BATCH_SIZE=1
HEAD_REDUCTION=max
CASCADE_FUNC="pow2"
# COMMENT="vanilla-truncate-right-half-ctx"
# COMMENT="vanilla"
# COMMENT="max-gen-adjust-quarter-ctx"
# COMMENT="max-gen-adjust"
# COMMENT=none
COMMENT=truncate
# COMMENT="half-ctx"

for i in "${!CASCADES[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu python cascade/main/llama_eval.py \
        --model $model \
        --job booksum \
        --method $method \
        --window $WINDOW \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $cascade_stride \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
