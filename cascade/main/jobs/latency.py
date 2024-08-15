import os
import torch
from tqdm import tqdm
import json
import time
import deepspeed
from cascade.main.jobs.pg19 import get_injection_policy


def job_latency(args, model, tokenizer, device):
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

    stride = 1
    for l in [2**i for i in range(12, 20)]:
        input_ids = torch.randn(1, 1024, 4096).cuda().repeat(1, l // 1024, 1).half()
        position_embeddings = m.rotary_emb(input_ids, torch.arange(l).cuda().view(1, -1))
        model = m.layers[0].self_attn

        if args.method != "vanilla":
            m.clear_caches()

        with torch.no_grad():
            if args.method == "sink":
                for i in range(0, input_ids.size(1), stride):
                    inp = input_ids[:, i:i + stride]
                    output = model(inp, use_cache=False)

                tic = time.perf_counter()
                for i in range(0, input_ids.size(1), stride):
                    inp = input_ids[:, i:i + stride]
                    output = model(inp, use_cache=False)

                torch.cuda.synchronize()
                print(f"latency for sink: len: {l}: {time.perf_counter() - tic}")

            else:
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                tic = time.perf_counter()
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                torch.cuda.synchronize()
                print(f"latency for vanilla: len: {l}: {time.perf_counter() - tic}")
