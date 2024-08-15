import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
from aim import Run
import numpy as np
from cascade.utils import MockRun
from cascade.dataset.passkey import Passkey


def get_numbers(s):
    lst = [c for c in s if c.isdigit()]
    out = ''.join(lst)[:5]
    if len(out) < 5:
        out += "A" * (5 - len(out))
    return out


def len_loc_str(len_loc):
    length, loc = len_loc
    return f"{length}-{loc}"


def job_passkey(args, model, tokenizer, device):
    if args.method != "vanilla":
        model.model.setup_caches(args.world_size, verbose=False)

    model = model.to(args.infer_dtype).cuda()
    m = model.model

    dataset = Passkey(tokenizer, batch_size=args.batch_size)

    stride = 4096
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

            guess_string = [get_numbers(s.strip()) for s in tokenizer.batch_decode(guesses)]
            # print(f"{guess_string=} {targets=}")
            guess_string, target_string = "".join(guess_string), "".join(targets)
            print(guess_string, target_string)

            for x, y in zip(guess_string, target_string):
                if x == y:
                    correct += 1
                count += 1

            len_loc = len_locs[0]  # they should all be the same in a single batch

            ll_str = len_loc_str(len_loc)
            if ll_str not in total_acc.keys():
                total_acc[ll_str] = [correct, count]
            else:
                total_acc[ll_str][0] += correct
                total_acc[ll_str][1] += count

            for k, v in total_acc.items():
                print(f"{k}: {v[0] / v[1]}")

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "sink":
        with open(
                f'./cache/llama_eval/ppl-passkey-{args.method}-sinks-{args.sinks}-window-{args.window}-cascade-{args.cascades}-{args.model}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}.json',
                'w') as f:
            json.dump(total_acc, f)


if __name__ == "__main__":
    out = get_numbers("some string 1 with a 3 few numbers 4")
    print(out)
