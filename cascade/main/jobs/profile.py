import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from cascade.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import argparse
import json
from transformers import TextStreamer
from aim import Run

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from cascade.utils import seed, get_bench, MockRun
import deepspeed
from cascade.models.modeling_llama import LlamaDecoderLayer
from cascade.models.qwen.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from third_party.hyper_attn.models.attention.modeling_chatglm_fast_attention import FastCoreAttention

from torch.profiler import profile, record_function, ProfilerActivity


def job_profile(args, model, tokenizer, device):
    stride = 1024
    model.model.setup_caches(args.world_size)
    model = model.to(args.infer_dtype).cuda()
    mdl = model.model

    nll_total = 0
    count_total = 0

    nll_individual = torch.zeros(100)
    count_individual = torch.zeros(100)
    step = 0

    dataset = PG19Streaming(tokenizer, batch_size=args.batch_size)

    nll_total, count_total = 0, 0
    j = 0
    x, y = dataset[0]
    input_ids = x.cuda()
    target_ids = y.cuda()

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True) as prof:
        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():

                    inputs = input_ids[:, i:i + stride]
                    targets = target_ids[:, i + 1:i + stride + 1]

                    if i == 0:
                        mdl.clear_caches()

                    inp = inputs
                    output = model(inp,
                                   use_cache=False,
                                   reset=i == 0,
                                   output_attentions=False)

                    logits = output.logits

                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    )

                    nll_total += _nll.sum()
                    count_total += (targets >= 0).sum().item()
                    pbar.set_description(
                        f"ppl: {(nll_total / count_total).exp()}")

                    l, u = j * args.batch_size, (j + 1) * args.batch_size
                    _nll = _nll.reshape(args.batch_size, -1)
                    nll_individual[l:u] += _nll.cpu().sum(dim=-1)
                    count_individual[l:u] += (targets >= 0).sum(dim=-1).cpu()
                    step += stride

                    book_idx = l + torch.arange(args.batch_size)
                    book_ppl = torch.exp(nll_individual[book_idx] /
                                         count_individual[book_idx])

                    stats = {}
                    for k in range(args.batch_size):
                        key = f"ppl-pg19-book-{l + k}"
                        val = book_ppl[k].item()
                        # only track items which have not reached EOS
                        if targets.reshape(-1)[k] >= 0:
                            stats[key] = val

                    print(f"{i=}")
                    # if i == 2 * stride:
                    break

    ppl = torch.exp(nll_total / count_total).item()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage",
                                    row_limit=500))
