# TimberAttention

## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python timber/models/sink_cache_cascade.py
```

## Run LongBench

```
cd third_party/LongBench-timber/

# edit run.sh to run the correct models/datasets
./run.sh
```

## TODO


## Note
