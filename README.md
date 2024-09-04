# Cascading KV Cache

# TODO:

- LongBench
    - qwen
    - llama

- MMLU
    - qwen
    - llama

- token selection ablation (PG19)
    - llama

- latency
    - compare attention latency of (decode, batch prompt, flash prompt)

- attention matrix plotting.

- longer context benchmarks

- pay digitalocean bill
- add more baselines
- note in new paper about beta 0.999
  - also note about the approximate score of flash attention?


## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python cascade/models/sink_cache_cascade.py
```

## RUN PG19

```
# set parameters for desired experiment in ./test-pg19.bash
./test-pg19.bash
```

## Run LongBench

```
cd third_party/LongBench-timber/

# edit run.sh to run the correct models/datasets
./run.sh
```

## Run MMLU

```
# set parameters for desired experiment in ./test-mmlu.bash
./test-mmlu.bash
```
