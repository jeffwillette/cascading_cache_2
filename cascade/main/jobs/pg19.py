import os
import torch
from cascade.dataset.pg19 import PG19Streaming
import numpy as np
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache
from cascade.utils.other import pad_targets


def job_ppl_pg19(args, model, tokenizer, device):
    stride = args.cascade_stride
    mdl = model.model

    # all_nll = []  # for hyper attention loop
    stats = {"nll-total": 0, "count-total": 0, "ppl-book": {}}
    dataset = PG19Streaming(tokenizer)

    past_key_values = None
    use_cache = False

    for j, (x, y) in enumerate(dataset):
        input_ids = x.cuda()
        target_ids = y.cuda()

        if args.method == "sink":
            use_cache = True

            # max_seq_len = [65536] * 4 + [8192] * 28
            # max_seq_len = [65536] * 4 + [16384] * 28
            # max_seq_len = [131072] * 2 + [8192] * 30
            # window = [m // args.cascades for m in max_seq_len]

            if "quarter-book" in args.comment:
                max_seq_len = int(2 ** int(np.log2(input_ids.size(1) / 4) // 1))
            elif "half-book" in args.comment:
                max_seq_len = int(2 ** int(np.log2(input_ids.size(1) / 2) // 1))
            else:
                max_seq_len = mdl.config._window

            if "llama" in args.model:
                # max_seq_len = min(max_seq_len, 32768)
                max_seq_len = min(max_seq_len, 65536)
            elif "qwen" in args.model:
                max_seq_len = min(max_seq_len, 16384)

            print(f"{max_seq_len=}")
            window = max_seq_len // args.cascades

            past_key_values = CascadingKVCache(
                window,
                num_sink_tokens=mdl.config._sinks,
                max_batch_size=mdl.config._batch_size,
                heads=mdl.config.num_key_value_heads // args.world_size,
                dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
                max_seq_len=max_seq_len,
                dtype=torch.float16,
                device=mdl.embed_tokens.weight.device,
                cascade_func=mdl.config._cascade_func,
                head_reduction=mdl.config._head_reduction,
                layers=len(mdl.layers),
            )
        elif args.method == "vanilla":
            max_seq_len = args.cascade_stride
        else:
            raise ValueError(f"unsupported method: {args.method=}")

        print(f"starting book: {input_ids.size()=}")
        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():

                    inputs = input_ids[:, i:i + stride]
                    targets = target_ids[:, i + 1:i + stride + 1]
                    targets = pad_targets(inputs, targets, ignore_index=-100)

                    output = model(inputs, use_cache=use_cache, past_key_values=past_key_values)
                    past_key_values = output.past_key_values
                    logits = output.logits

                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    )

                    nll, count = _nll.sum().item(), (targets >= 0).sum().item()

                    stats["nll-total"] += nll
                    stats["count-total"] += count

                    pbar.set_description(f"ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")

                    if j not in stats['ppl-book'].keys():
                        stats['ppl-book'][j] = []
                    stats['ppl-book'][j].append((nll, count, max_seq_len))

        del past_key_values, logits, targets, inputs, output, input_ids, target_ids
        torch.cuda.empty_cache()

    print(f"final ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")
    os.makedirs('./cache/llama_eval/pg19/', exist_ok=True)
    f = f'./cache/llama_eval/pg19/{args.method}-{args.model}-{args.comment}-window-{args.window}-' + \
        f'cascades-{args.cascades}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}-' + \
        f'cascade-stride-{args.cascade_stride}-homogeneous-heads-{args.homogeneous_heads}-comment-{args.comment}.json'

    with open(f, 'w') as f:
        json.dump(stats, f)
