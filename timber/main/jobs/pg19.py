import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from timber.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import argparse, json
from transformers import TextStreamer
from aim import Run

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench, MockRun
import deepspeed
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer


def job_ppl_pg19(args, model, tokenizer, device):
    # model.model.setup_caches()
    # model = torch.compile(model, mode="max-autotune", fullgraph=False)
    model.model.setup_caches(args.world_size)
    model = model.to(args.infer_dtype)
    model = deepspeed.init_inference(model,
                                     tensor_parallel={"tp_size": 4},
                                     replace_with_kernel_inject=False,
                                     dtype=args.infer_dtype,
                                     injection_policy={
                                         LlamaDecoderLayer: (
                                             'mlp.down_proj',
                                             'self_attn.o_proj',
                                         )
                                     })
    # model = torch.compile(model, mode="max-autotune", fullgraph=False)

    run = MockRun()
    if args.local_rank == 0:
        run = Run(experiment=f"{args.method}-pg19"
                  ) if not args.dev_run else MockRun()
        dataset = "pg19"
        run["hparams"] = {
            "job": "ppl",
            "method": args.method,
            "dataset": dataset,
            "sinks": args.sinks,
            "cascades": args.cascades,
            "window": args.window,
            "slots": args.slots,
            "model": args.model,
            "cascade-func": args.cascade_func,
            "comment": args.comment,
        }

    nll_total = 0
    count_total = 0

    nll_individual = torch.zeros(100)
    count_individual = torch.zeros(100)
    step = 0

    dataset = PG19Streaming(tokenizer, batch_size=args.batch_size)

    for j, (x, y) in enumerate(dataset):
        input_ids = x.cuda()
        target_ids = y.cuda()
        print(
            f"starting batch of books: {input_ids.size()=} {target_ids.size()=}"
        )
        with tqdm(range(x.size(1) - 1), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():
                    inp = input_ids[:, i:i + 1]
                    # use cache false means to use static cascading cache inside the model
                    output = model(inp, use_cache=False, reset=i == 0)

                    logits = output.logits[:, -1:]
                    targets = target_ids[:, i + 1:i + 2]
                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    )

                    nll_total += _nll.sum()
                    count_total += (targets >= 0).sum().item()

                    l, u = j * args.batch_size, (j + 1) * args.batch_size
                    nll_individual[l:u] += _nll.cpu()
                    count_individual[l:u] += (targets >= 0).sum(dim=-1).cpu()
                    step += 1

                    if step % 100 == 0:
                        ppl = torch.exp(nll_total / count_total).item()
                        run.track(ppl,
                                  name="ppl-pg19",
                                  step=step,
                                  context={"subset": "test"})
                        pbar.set_description(f"{ppl=:.6f}")

                        book_idx = l + torch.arange(args.batch_size)
                        book_ppl = torch.exp(nll_individual[book_idx] /
                                             count_individual[book_idx])

                        stats = {}
                        for k in range(args.batch_size):
                            key = f"ppl-pg19-book-{l + k}"
                            val = book_ppl[k].item()
                            # only track items which have not reached EOS
                            if targets.view(-1)[k] >= 0:
                                stats[key] = val

                        run.track(stats, step=i, context={"subset": "test"})

    ppl = torch.exp(nll_total / count_total).item()

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "sink":
        with open(
                f'./cache/llama_eval/pg19-{args.method}-sinks-{args.sinks}-window-{args.window}.json',
                'w') as f:
            json.dump({'ppl': ppl}, f)
    else:
        raise ValueError(
            f"methods other than sink are not supported: {args.method=}")

    print(f'PPL: {ppl:.4f}')
