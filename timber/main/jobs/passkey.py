import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
from aim import Run
import numpy as np
from timber.utils import MockRun
from timber.dataset.passkey import Passkey


def job_passkey(args, model, tokenizer, device):
    model.model.setup_caches(args.world_size)
    model = model.to(args.infer_dtype).cuda()
    model = torch.compile(model, mode="max-autotune", fullgraph=False)

    dataset = Passkey(tokenizer, batch_size=args.batch_size)

    total_acc, total_acc_per_token = [], []
    for j, (input_ids, target_ids) in enumerate(tqdm(dataset, ncols=150)):
        input_ids = input_ids.cuda()
        target_ids = target_ids.cuda()

        model.model.clear_caches()
        correct, count = 0, 0
        correct_per_token, count_per_token = 0, 0

        with torch.no_grad():
            with tqdm(range(input_ids.size(1)), ncols=150) as pbar2:
                for i in pbar2:
                    inp = input_ids[:, i:i + 1]
                    output = model(inp, use_cache=False, reset=i == 0)

                guesses, terminated, i = [], [False] * target_ids.size(0), 0
                while not all(terminated):
                    pred = output.logits[:, -1:].argmax(dim=-1)
                    guesses += [pred]
                    output = model(pred, use_cache=False, reset=False)

                    for j in range(output.logits.size(0)):
                        if output.logits[j,
                                         0].argmax() == tokenizer.eos_token_id:
                            terminated[j] = True

                    i += 1
                    if i == 100:
                        break

                guesses = torch.cat(guesses, dim=-1)

                guess_string = tokenizer.batch_decode(guesses)
                target_string = tokenizer.batch_decode(target_ids)
                print(f"{guess_string=} {target_string=}")

                correct = sum(
                    [t in g for t, g in zip(target_string, guess_string)])
                count += len(guess_string)

                for (t, g) in zip(target_string, guess_string):
                    for n in t:
                        count_per_token += 1
                        if n in g:
                            correct_per_token += 1

                acc = correct / count
                acc_per_token = correct_per_token / count_per_token

                print(f"{acc=} {acc_per_token=}")
                total_acc.append(acc)
                total_acc_per_token.append(acc_per_token)
                # total_acc_per_token.append(acc_per_token)

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "sink":
        with open(
                f'./cache/llama_eval/ppl-passkey-{args.method}-sinks-{args.sinks}-window-{args.window}-cascade-{args.cascades}-{args.model}.json',
                'w') as f:
            json.dump(
                {
                    'accuracy': total_acc,
                    "acc_per_token": total_acc_per_token
                }, f)
    else:
        raise NotImplementedError(f"{args.method} not implemented")
