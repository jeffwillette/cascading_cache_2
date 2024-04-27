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
from timber.utils import seed, get_bench


class MockRun:

    def __init__(self, *args, **kwargs):
        pass

    def track(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass


def job_ppl_pg19(args, model, tokenizer, device):
    model.model.setup_caches()
    run = Run(
        experiment=f"{args.method}-pg19") if not args.dev_run else MockRun()
    dataset = "wikitext"
    run["hparams"] = {
        "job": "ppl",
        "method": args.method,
        "dataset": dataset,
        "sinks": args.sinks,
        "cascades": args.cascades,
        "window": args.window,
        "slots": args.slots,
    }

    nll = 0
    count = 0
    batch_size = 25
    dataset = PG19Streaming(tokenizer, batch_size=batch_size)

    for x, y in dataset:
        input_ids = x.to(device)
        target_ids = y.to(device)
        print(
            f"starting batch of books: {input_ids.size()=} {target_ids.size()=}"
        )
        with tqdm(range(x.size(1) - 1)) as pbar:
            for i in pbar:
                with torch.no_grad():
                    inp = input_ids[:, i:i + 1]
                    # use cache false means to use static cascading cache inside the model
                    output = model(inp, use_cache=False)

                    logits = output.logits[:, -1:]
                    targets = target_ids[:, i + 1:i + 2]
                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    nll += _nll
                    count += (targets >= 0).sum().item()

                    if i % 1 == 0:
                        ppl = torch.exp(nll / count).item()
                        run.track(ppl,
                                  name="ppl-pg19",
                                  step=i,
                                  context={"subset": "test"})
                        pbar.set_description(f"{ppl=:.6f}")

    ppl = torch.exp(nll / count).item()

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "umbc":
        with open(
                f'./cache/llama_eval/pg19-{args.method}-sinks-{args.sinks}-window-{args.window}-slots-{args.slots}.json',
                'w') as f:
            json.dump({'ppl': ppl}, f)
    elif args.method == "sink":
        with open(
                f'./cache/llama_eval/pg19-{args.method}-sinks-{args.sinks}-window-{args.window}.json',
                'w') as f:
            json.dump({'ppl': ppl}, f)
    else:
        with open(f'./cache/llama_eval/pg19-{args.method}.json', 'w') as f:
            json.dump({'ppl': ppl}, f)

    print(f'PPL: {ppl:.4f}')
