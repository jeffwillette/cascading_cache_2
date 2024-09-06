#!/bin/bash

WINDOW=16384
CASCADES=4
SINKS=4
BATCH_SIZE=1
HEAD_REDUCTION=(max)
CASCADE_FUNC="pow2"
GPUS=(5)
# MODEL=llama3.1-8b
MODEL=qwen2-7b
# MODEL=llama3.1-70b
# MODEL=llama3.1-70b-instruct-gptq-int4
METHOD=sink
# COMMENT="different-cache-sizes-65k-4-8k-28"
# COMMENT="different-cache-sizes-65k-4-16k-28"
# COMMENT="different-cache-sizes-131k-2-8k-30"
# COMMENT="window-quarter-book-score-correct-32768all"
COMMENT="16384all"
# COMMENT="window-half-book"
# COMMENT="plain"
CASCADE_STRIDE=1024

# MAIN PG19 experiment code
for i in "${!HEAD_REDUCTION[@]}";
do 
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPUS[$i]} python cascade/main/llama_eval.py \
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
        --batch_size $BATCH_SIZE
        sleep 1
done
