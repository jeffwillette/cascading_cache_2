import argparse
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ArgsType:
    model: Literal['llama7b', 'llama13b', 'qwen7b', 'qwen14b'] = 'llama32k'
    job: Literal['ppl', 'ppl-memory', 'mmlu'] = 'ppl'
    method: Literal['none', 'sink'] = 'sink'
    stride: int = -1
    lora_r: int = 32
    checkpoint: Optional[str] = None
    count: int = 100
    block_size_q: int = 32
    block_size_k: int = 2
    batch_size: int = 1
    k: int = 512
    chunk: int = 16


def eval_args(
    default_model='llama32k',
    default_job='ppl',
) -> ArgsType:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--job', type=str, default=default_job)
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--cascade_stride', type=int, default=1024)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--window', default=256, type=int)
    parser.add_argument('--chunk', default=32, type=int)
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument('--sinks', default=0, type=int)
    parser.add_argument('--cascades', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--cascade_func', type=str, default='pow2')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--head_reduction', type=str, default='mean')
    parser.add_argument('--homogeneous_heads', action='store_true')
    parser.add_argument('--do_og_pos', action='store_true')
    parser.add_argument('--dev_run', action='store_true')

    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print(args)
    return args
