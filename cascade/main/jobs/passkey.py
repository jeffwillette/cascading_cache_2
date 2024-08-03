import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
from aim import Run
import numpy as np
from cascade.utils import MockRun
from cascade.dataset.passkey import Passkey
import deepspeed
from cascade.main.jobs.pg19 import get_injection_policy


def get_numbers(s):
    lst = [c for c in s if c.isdigit()]
    return ''.join(lst)


def job_passkey(args, model, tokenizer, device):
    if args.method != "vanilla":
        model.model.setup_caches(args.world_size, verbose=False)

    if args.world_size > 1:
        model = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": args.world_size},
            replace_with_kernel_inject=False,
            dtype=args.infer_dtype,

            injection_policy=get_injection_policy(args.model),
        )
        m = model.module.model
    else:
        model = model.to(args.infer_dtype).cuda()
        # model = torch.compile(model, mode="max-autotune", fullgraph=False)
        m = model.model

    dataset = Passkey(tokenizer, batch_size=args.batch_size)

    stride = 1024
    total_acc = {}
    for j, (input_ids, targets, len_locs) in enumerate(tqdm(dataset, ncols=150)):
        input_ids = input_ids.cuda()

        if args.method != "vanilla":
            m.clear_caches()

        correct, count = 0, 0

        with torch.no_grad():
            if args.method == "sink":
                with tqdm(range(0, input_ids.size(1), stride), ncols=150) as pbar2:
                    for i in pbar2:
                        inp = input_ids[:, i:i + stride]
                        output = model(inp, use_cache=False, reset=i == 0)

                # print(f"{inp.size()=} {input_ids.size()=}")
                guesses, i = [], 0
                for i in range(100):
                    pred = output.logits[:, -1:].argmax(dim=-1)
                    guesses += [pred]
                    output = model(pred, use_cache=False, reset=False)

                guesses = torch.cat(guesses, dim=-1)

            else:
                output = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    min_new_tokens=100,
                    do_sample=False,
                    num_beams=1,
                )
                guesses = output[:, input_ids.shape[1]:]

            guess_string = [get_numbers(s.strip())[:5] for s in tokenizer.batch_decode(guesses)]
            # print(f"{guess_string=} {targets=}")
            guess_string, target_string = "".join(guess_string), "".join(targets)
            print(guess_string, target_string)

            for x, y in zip(guess_string, target_string):
                if x == y:
                    correct += 1
                count += 1

            len_loc = len_locs[0]  # they should all be the same in a single batch

            if len_loc not in total_acc.keys():
                total_acc[len_loc] = [correct, count]
            else:
                total_acc[len_loc][0] += correct
                total_acc[len_loc][1] += count

            for k, v in total_acc.items():
                print(f"{k}: {v[0] / v[1]}")

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "sink":
        with open(
                f'./cache/llama_eval/ppl-passkey-{args.method}-sinks-{args.sinks}-window-{args.window}-cascade-{args.cascades}-{args.model}.json',
                'w') as f:
            json.dump(total_acc, f)
    else:
        raise NotImplementedError(f"{args.method} not implemented")
