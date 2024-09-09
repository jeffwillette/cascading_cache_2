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
# WINDOW=(32768 65536 32768 65536)
# CASCADES=(4 4 1 1)
# # TODO: UNDO THE FOLLOWING THREE LINES< THEY ARE TEMPORARY
WINDOW=(16384 16384)
CASCADES=(1 1)
MODEL=(llama3.1-8b qwen2-7b)



SINKS=4
BATCH_SIZE=1
CASCADE_FUNC="pow2"
COMMENT="none"
CASCADE_STRIDE=1024

# MAIN PG19 experiment code
for i in "${!WINDOW[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU python cascade/main/llama_eval.py \
        --model ${MODEL[$i]} \
        --job ppl-pg19 \
        --method $METHOD \
        --window ${WINDOW[$i]} \
        --sinks $SINKS \
        --cascades ${CASCADES[$i]} \
        --cascade_func $CASCADE_FUNC \
        --cascade_stride $CASCADE_STRIDE \
        --head_reduction $HEAD_REDUCTION \
        --comment $COMMENT \
        --batch_size $BATCH_SIZE
        sleep 1
done
