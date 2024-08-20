import os
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache
from cascade.utils.other import pad_targets


datasets = {
    "wikitext": "wikitext-103-raw-v1",
    "openwebtext": "openwebtext",
}


def job_ppl(args, model, tokenizer, device):
    for dataset in datasets.keys():
        args.dataset = dataset
        job_ppl_inner(args, model, tokenizer, device)


def job_ppl_inner(args, model, tokenizer, device):
    mdl = model.model
    dataset = args.dataset

    cache_dir = f'./cache/llama_eval/{args.dataset}'
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = f'{cache_dir}/{args.model}-tokenized.pth'
    if not os.path.exists(cache_path):
        test = load_dataset(dataset, "wikitext-103-raw-v1", split="test")
        print(test)
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
        torch.save(encodings, cache_path)
    else:
        encodings = torch.load(cache_path)

    seq_len = encodings.size(1)
    stride = args.cascade_stride

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
            heads=mdl.config.num_key_value_heads // args.world_size,
            dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
            max_seq_len=max_seq_len,
            dtype=torch.float16,
            device=mdl.embed_tokens.weight.device,
            cascade_func=mdl.config._cascade_func,
            head_reduction=mdl.config._head_reduction,
            layers=len(mdl.layers),
        )

    stats = {"nll-total": 0, "count-total": 0, "nll-by-step": []}

    with tqdm(range(0, seq_len, stride)[:args.count]) as pbar:
        for i in pbar:
            input_ids = encodings[:, i: i + stride].to(device)
            target_ids = input_ids.clone()
            target_ids = target_ids[:, i + 1: i + 1 + stride]

            target_ids = pad_targets(input_ids, target_ids, ignore_index=-100)

            with torch.no_grad():
                output = model(
                    input_ids,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
                past_key_values = output.past_key_values

                nll = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, output.logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=-100,  # from pad_targets function
                    reduction="none",
                )

                count = (target_ids >= 0).sum().item()
                stats["nll-total"] += nll.sum().item()
                stats["count-total"] += count
                stats["nll-by-step"].append((count, nll.sum().item()))

            ppl = np.exp(stats["nll-total"] / stats["count-total"])
            pbar.set_description(f"ppl: {ppl:.3f}")

    fl = f'{cache_dir}/{args.method}-{args.model}-{args.comment}-window-{args.window}-' + \
        f'cascades-{args.cascades}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}-' + \
        f'cascade-stride-{args.cascade_stride}.json'

    with open(fl, 'w') as f:
        json.dump(stats, f)

    print(f'PPL: {ppl:.4f}')
