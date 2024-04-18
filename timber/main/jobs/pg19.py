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


def job_ppl_pg19(args, model, tokenizer, device):
    run = Run(experiment=f"{args.method}-pg19")
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

    encodings, targets = PG19Streaming(tokenizer)[0]

    nll = 0
    count = 0
    past_key_values = None
    input_ids = encodings.to(device)
    target_ids = targets.to(device)
    with tqdm(range(encodings.size(1) - 1)) as pbar:
        for i in pbar:
            with torch.no_grad():
                # model.model.model.sse.post_forward_mbc_cleanup()
                mdl = model.model if args.lora_r == 0 else model.model.model

                if args.method == "umbc":
                    for lyr in mdl.layers:
                        lyr.self_attn.sse.post_forward_mbc_cleanup()

                inp = input_ids[:, i:i + 1]
                output = model(
                    inp,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = output.past_key_values

                logits = output.logits[:, -1:]
                targets = target_ids[:, i + 1:i + 2]
                _nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                nll += _nll
                count += 1

                if i % 10 == 0:
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
