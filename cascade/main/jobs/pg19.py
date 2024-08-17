import os
import torch
from cascade.dataset.pg19 import PG19Streaming
import numpy as np
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache


def job_ppl_pg19(args, model, tokenizer, device):
    stride, world_size = args.cascade_stride, 1
    mdl = model.model

    # all_nll = []  # for hyper attention loop
    stats = {"nll-total": 0, "count-total": 0, "ppl-book": {}}
    dataset = PG19Streaming(tokenizer)

    past_key_values = None
    use_cache = False
    if args.method == "sink":
        use_cache = True

        window = mdl.config._window // mdl.config._cascades
        max_seq_len = mdl.config._window

        past_key_values = CascadingKVCache(
            window,
            num_sink_tokens=mdl.config._sinks,
            max_batch_size=mdl.config._batch_size,
            heads=mdl.config.num_key_value_heads // world_size,
            dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
            max_seq_len=max_seq_len,
            dtype=torch.float16,
            device=mdl.embed_tokens.weight.device,
            cascade_func=mdl.config._cascade_func,
            head_reduction=mdl.config._head_reduction,
            layers=len(mdl.layers),
        )

    for j, (x, y) in enumerate(dataset):
        input_ids = x.cuda()
        target_ids = y.cuda()

        if past_key_values is not None:
            past_key_values.reset()

        print(f"starting book: {input_ids.size()=}")
        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():

                    inputs = input_ids[:, i:i + stride]
                    targets = target_ids[:, i + 1:i + stride + 1]

                    # should only happen if they are off by 1 at the very end
                    if targets.size(1) < inputs.size(1):
                        target_pad = torch.full(
                            (inputs.size(0), inputs.size(1) - targets.size(1)),
                            -100,
                            device=inputs.device,
                            dtype=inputs.dtype)

                        targets = torch.cat((targets, target_pad), dim=-1)

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

                    step = min(stride, count)
                    stats["nll-total"] += nll
                    stats["count-total"] += count

                    pbar.set_description(f"ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")

                    if j not in stats['ppl-book'].keys():
                        stats['ppl-book'][j] = []
                    stats['ppl-book'][j].append((step, nll, count))

    print(f"final ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")
    os.makedirs('./cache/llama_eval/pg19/', exist_ok=True)
    f = f'./cache/llama_eval/pg19/{args.method}-{args.model}-{args.comment}-window-{args.window}-' + \
        f'cascades-{args.cascades}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}.json'

    with open(f, 'w') as f:
        json.dump(stats, f)
