import os
import torch
from cascade.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache
import numpy as np
from cascade.utils.other import pad_targets


def job_attn_matrix(args, model, tokenizer, device):
    stride = args.cascade_stride
    mdl = model.model

    assert args.method == "sink", f"attention matrix plotting supported for sink models only: got: {args.method}"

    # all_nll = []  # for hyper attention loop
    stats = {"nll-total": 0, "count-total": 0, "ppl-book": {}}
    dataset = PG19Streaming(tokenizer)

    past_key_values = None
    use_cache = False

    path = "cache/pg19/stats.json"
    if not os.path.exists(path):
        raise ValueError("run pg19 dataset file as __main__ to generate stats")

    with open(path, "r") as fl:
        ds_stats = json.load(fl)
        ds_stats = {int(k): v for k, v in ds_stats.items()}

    def populate(pkv, om, first, layers, heads):
        for lyr in range(layers):
            for head in range(heads):
                og_pos = pkv._attn_plot_og_pos[lyr][0, head]
                attn = pkv._attn_plot_attn_scores[lyr][0, head]

                og_pos, attn = og_pos.cpu(), attn.cpu()

                sink_attn, kv_attn, stride_attn = attn[:, :64], attn[:, 64:-stride], attn[:, -stride:]
                if first:
                    om[lyr, head, i:i + stride, i:i + stride] = stride_attn
                else:
                    om[lyr, head, i:i + stride, :64] = sink_attn
                    om[lyr, head, i:i + stride].scatter_(1, og_pos.view(1, -1).repeat(stride, 1), kv_attn)

                    start = i
                    # print(f"{start=} {start + stride=}")
                    stride_pos = torch.arange(start, start + stride, device=stride_attn.device).view(1, -1).repeat(stride, 1)

                    om[lyr, head, i:i + stride].scatter_(1, stride_pos, stride_attn)

    for j, (x, _) in enumerate(dataset):
        input_ids = x.cuda()

        if args.method == "sink":
            use_cache = True

            max_seq_len = mdl.config._window
            max_seq_len = min(max_seq_len, mdl.config.max_position_embeddings // 2)

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
            past_key_values.force_eager = True
            past_key_values.plot_attn = True
        else:
            raise ValueError(f"unsupported method for attn matrix plotting: {args.method=}")

        out_matrix = torch.zeros(32, 8, 8192, 8192, dtype=torch.float16)
        layers, heads = 32, 8
        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                print(f"{i=} {i + stride=}")
                if i + stride > 8192:
                    break

                with torch.no_grad():

                    inputs = input_ids[:, i:i + stride]

                    output = model(inputs, use_cache=use_cache, past_key_values=past_key_values)
                    past_key_values = output.past_key_values
                    populate(past_key_values, out_matrix, i == 0, layers, heads)

        with torch.no_grad():
            for lyr in range(layers):
                for head in range(heads):
                    m = out_matrix[lyr, head].clone()
                    # print(f"{m.size()=} {m.dtype=} {m.untyped_storage()=} {m.data_ptr()=}")
                    torch.save(m, f"plots/attention-matrix/cascade-{args.cascades}-layer-{lyr}-head-{head}.pt")

        break

        del past_key_values, inputs, output, input_ids
        torch.cuda.empty_cache()
