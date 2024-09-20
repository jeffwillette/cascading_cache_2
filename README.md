# Cascading KV Cache

# TODO:

- Passkey
- LongBench
- MMLU

- PG19
  - token selection ablation (use scores vs no scores) (PG19)

- longer context benchmarks
  - RULER?

- attention matrix plotting.
  - plot new ones with eager fill to see if there is any difference.

- add more baselines
  - snapKV booksum (running)
  - hyper attention pg19
  - h2o booksum?
  - bigbird booksum (running), pg19, 

- note in new paper about beta 0.999
  - also note about the approximate score of flash attention?
  - note about LLM lost in the middle paper (cite it and give inspiration in intro)


## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python cascade/models/sink_cache_cascade.py
```

## Run Passkey

For passkey, batch size must evenly divide 20 (1, 2, 4, 5, 10, 20)

```
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -w [WINDOW SIZE] -c [CASCADE NUMBER] -b [BATCH SIZE]

./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -w [WINDOW SIZE] -c 1 -b 2
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -w [WINDOW SIZE] -c 8 -b 2
```

## RUN PG19

```
# set parameters for desired experiment in ./test-pg19.bash
./test-pg19.bash
```

## Run LongBench

```
cd third_party/LongBench-timber/

./run.sh - m [MODEL] -d [METHOD] -g [GPU INDEX]

./run.sh -m llama3.1-8b-instruct -d sink -g [GPU INDEX]
./run.sh -m llama3.1-8b-instruct -d vanilla -g [GPU INDEX]
./run.sh -m qwen2-7b-instruct -d sink -g [GPU INDEX]
./run.sh -m qwen2-7b-instruct -d vanilla -g [GPU INDEX]

```

## Run MMLU

```
# set parameters for desired experiment in ./test-mmlu.bash
./test-mmlu.bash
```
