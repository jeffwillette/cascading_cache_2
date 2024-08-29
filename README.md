# Cascading KV Cache

# TODO:

- add more baselines and get final results for experiments
- re-run all other experiments to get new results with llama3.1 and qwen2
- note in new paper about beta 0.995

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
