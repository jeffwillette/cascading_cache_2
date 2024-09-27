import os
import torch
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache
from cascade.models.offloaded_cache import OffloadedCache
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
    savedir = './cache/llama_eval/passkey'
    os.makedirs(savedir, exist_ok=True)

    model = model.to(args.infer_dtype).cuda()
    m = model.model

    past_key_values = None
    if args.method != "vanilla":
        max_seq_len = m.config._window
        print(f"{max_seq_len=}")
        window = max_seq_len // args.cascades

        past_key_values = CascadingKVCache(
            window,
            num_sink_tokens=m.config._sinks,
            max_batch_size=m.config._batch_size,
            heads=m.config.num_key_value_heads // args.world_size,
            dim=m.config.hidden_size // m.config.num_attention_heads,
            max_seq_len=max_seq_len,
            dtype=torch.float16,
            device=m.embed_tokens.weight.device,
            cascade_func=m.config._cascade_func,
            head_reduction=m.config._head_reduction,
            layers=len(m.layers),
        )

    pk_type = "1M"
    if args.comment == "262K":
        pk_type = "262K"

    dataset = Passkey(tokenizer, batch_size=args.batch_size, max_tokens=pk_type)

    stride = args.cascade_stride

    savepath = f"{savedir}/{args.method}-sinks-{args.sinks}-window-" + \
        f"{args.window}-cascade-{args.cascades}-{args.model}-head-reduction-" + \
        f"{args.head_reduction}-cascade-func-{args.cascade_func}.json"

    total_acc = {}
    if os.path.exists(savepath):
        with open(savepath, 'r') as f:
            total_acc = json.load(f)

    for j, (input_ids, targets, len_locs) in enumerate(tqdm(dataset, ncols=150)):
        len_loc = len_locs[0]  # they should all be the same in a single batch

        ll_str = len_loc_str(len_loc)

        correct, count = 0, 0
        if ll_str not in total_acc.keys():
            total_acc[ll_str] = [correct, count]

        if total_acc[ll_str][1] == 100:
            print(f"skipping: {ll_str} because it has already been done")
            continue

        input_ids = input_ids.cuda()

        if args.method != "vanilla":
            past_key_values.reset(verbose=False)
        elif args.method == "vanilla":
            past_key_values = OffloadedCache()
        else:
            raise NotImplementedError(f"passkey pkv for {args.method=} is not implemented")

        with torch.no_grad():
            if args.method == "sink":
                with tqdm(range(0, input_ids.size(1) - 1, stride), ncols=150) as pbar:
                    for i in pbar:
                        # print(f"stride step: {i}")
                        inputs = input_ids[:, i:i + stride]
                        output = model(inputs, use_cache=True, past_key_values=past_key_values)
                        past_key_values = output.past_key_values
            elif args.method == "vanilla":
                output = model(input_ids, use_cache=True, past_key_values=past_key_values)
                past_key_values = output.past_key_values
            else:
                raise NotImplementedError(f"passkey not implemented for {args.method=}")

            # print(f"{inp.size()=} {input_ids.size()=}")
            guesses, i = [], 0
            for i in range(50):
                pred = output.logits[:, -1:].argmax(dim=-1)
                guesses.append(pred.cpu())
                output = model(pred, use_cache=True, past_key_values=past_key_values)

            guesses = torch.cat(guesses, dim=-1)
            # print(f"{guesses=}\n{guesses.size()=}")
            output: str = tokenizer.batch_decode(
                guesses.data.cpu(),
                skip_special_tokens=True,
            )
            # print(f"{output=}")

            guess_string = [get_numbers(s.strip()) for s in output]
            print(f"{guess_string=} {targets=}")
            guess_string, target_string = "".join(guess_string), "".join(targets)
            # print(guess_string, target_string)

            for x, y in zip(guess_string, target_string):
                if x == y:
                    correct += 1
                count += 1

            total_acc[ll_str][0] += correct
            total_acc[ll_str][1] += count

            for k, v in total_acc.items():
                print(f"{k}: {v[0] / v[1]}")

        if total_acc[ll_str][1] == 100:
            with open(savepath, 'w') as f:
                json.dump(total_acc, f)


if __name__ == "__main__":
    out = get_numbers("some string 1 with a 3 few numbers 4")
    print(out)
