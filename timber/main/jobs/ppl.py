import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse, json
from transformers import TextStreamer
from aim import Run

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench


def job_ppl(args, model, tokenizer, device):
    run = Run(experiment=args.method)
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

    os.makedirs('./cache', exist_ok=True)
    cache_path = './cache/llama_eval.pth'
    if not os.path.exists(cache_path):
        # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test = load_dataset(dataset, "wikitext-103-raw-v1", split="test")
        # test = load_dataset("openwebtext", split="train")
        print(test)
        encodings = tokenizer("\n\n".join(test["text"]),
                              return_tensors="pt").input_ids
        torch.save(encodings, cache_path)
    else:
        encodings = torch.load(cache_path)

    max_length = model.config.max_position_embeddings
    # max_length = stride = args.stride if args.stride > 0 else model.config.max_position_embeddings
    seq_len = encodings.size(1)
    max_length = stride = seq_len

    nlls = []
    prev_end_loc = 0
    with tqdm(range(0, seq_len, stride)[:args.count]) as pbar:
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            if args.method in ["umbc", "sink"]:
                with torch.no_grad():
                    # model.model.model.sse.post_forward_mbc_cleanup()
                    mdl = model.model if args.lora_r == 0 else model.model.model

                    if args.method == "umbc":
                        for lyr in mdl.layers:
                            lyr.self_attn.sse.post_forward_mbc_cleanup()

                    rng = input_ids.size(1) - 1
                    past_key_values = None
                    with tqdm(range(rng)) as pbar2:
                        for i in pbar2:
                            inp = input_ids[:, i:i + 1]
                            output = model(
                                inp,
                                use_cache=True,
                                past_key_values=past_key_values,
                            )
                            past_key_values = output.past_key_values

                            logits = output.logits[:, -1:]
                            targets = target_ids[:, i + 1:i + 2]
                            nll = torch.nn.functional.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1),
                            )
                            nlls += [nll.cpu()]

                            if i % 10 == 0:
                                ppl = torch.exp(
                                    torch.stack(nlls).mean()).item()
                                run.track(ppl,
                                          name="ppl",
                                          step=i,
                                          context={"subset": "test"})
                                pbar2.set_description(f"{ppl=:.6f}")

            else:
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood.cpu())

            prev_end_loc = end_loc
            ppl = torch.exp(torch.stack(nlls).mean()).item()
            pbar.set_description(f"ppl: {ppl:.3f}")

            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean()).item()

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "umbc":
        with open(
                f'./cache/llama_eval/ppl-{args.method}-sinks-{args.sinks}-window-{args.window}-slots-{args.slots}.json',
                'w') as f:
            json.dump({'ppl': ppl}, f)
    elif args.method == "sink":
        with open(
                f'./cache/llama_eval/ppl-{args.method}-sinks-{args.sinks}-window-{args.window}.json',
                'w') as f:
            json.dump({'ppl': ppl}, f)
    else:
        with open(f'./cache/llama_eval/ppl-{args.method}.json', 'w') as f:
            json.dump({'ppl': ppl}, f)

    print(f'PPL: {ppl:.4f}')
