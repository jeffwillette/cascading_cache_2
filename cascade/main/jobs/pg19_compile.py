import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from cascade.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import argparse
import json
from transformers import TextStreamer
# from aim import Run

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from cascade.utils import seed, get_bench, MockRun
import deepspeed
from cascade.models.modeling_llama import LlamaDecoderLayer
from cascade.models.qwen.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from third_party.hyper_attn.models.attention.modeling_chatglm_fast_attention import FastCoreAttention


def get_injection_policy(model_id):
    if "llama" in model_id.lower():
        return {
            LlamaDecoderLayer: (
                'mlp.down_proj',
                'self_attn.o_proj',
            )
        }
    elif "qwen" in model_id.lower():
        return {
            Qwen2DecoderLayer: (
                'mlp.down_proj',
                'self_attn.o_proj',
            ),
        }
    else:
        raise ValueError()


def job_ppl_pg19_compile(args, model, tokenizer, device):
    stride = 512
    args.graph = False
    if args.world_size == 1 and args.graph:
        model.model.setup_caches(args.world_size)
        model = model.to(args.infer_dtype).cuda()
        # model = torch.compile(model, backend="cudagraphs")

        # ==================================================================
        inp = torch.randint(100, size=(args.batch_size, stride)).cuda()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(s):
                for i in range(3):
                    output = model(
                        inp,
                        use_cache=False,
                        reset=i == 0,
                        output_attentions=False,
                    )
                    logits = output.logits

        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        with torch.no_grad():
            with torch.cuda.graph(g):
                output = model(
                    inp,
                    use_cache=False,
                    reset=False,
                    output_attentions=False,
                )
                logits = output.logits

        # ================================================================

    else:
        model.model.setup_caches(args.world_size)
        model = model.to(args.infer_dtype).cuda()
        # model = deepspeed.init_inference(
        #     model,
        #     tensor_parallel={"tp_size": args.world_size},
        #     replace_with_kernel_inject=False,
        #     dtype=args.infer_dtype,
        #     injection_policy=get_injection_policy(args.model),
        # )

    run = MockRun()
    if args.local_rank == 0:
        # run = Run(experiment=f"{args.method}-pg19"
        #           ) if not args.dev_run else MockRun()

        run = MockRun()
        dataset = "pg19"
        run["hparams"] = {
            "job": "ppl",
            "method": args.method,
            "dataset": dataset,
            "sinks": args.sinks,
            "cascades": args.cascades,
            "window": args.window,
            "model": args.model,
            "head_reduction": args.head_reduction,
            "cascade_func": args.cascade_func,
            "comment": args.comment,
        }

    nll_total = 0
    count_total = 0

    all_nll = []  # for hyper attention loop
    nll_individual = torch.zeros(100)
    count_individual = torch.zeros(100)
    step = 0

    dataset = PG19Streaming(tokenizer, batch_size=args.batch_size)

    nll_total, count_total = 0, 0
    for j, (x, y) in enumerate(dataset):
        input_ids = x.cuda()
        target_ids = y.cuda()

        print(
            f"starting batch of books: {input_ids.size()=} {target_ids.size()=}"
        )

        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():
                    # inp = input_ids[:, i:i + 1]
                    # use cache false means to use static cascading cache inside the model

                    if args.world_size == 1 and args.graph:
                        if i == 0:
                            model.model.clear_caches()

                        inputs = input_ids[:, i:i + stride]
                        if inputs.size(1) < stride:
                            pad = torch.zeros(inputs.size(0),
                                              stride - inputs.size(1),
                                              device=inputs.device,
                                              dtype=inputs.dtype)

                            target_pad = torch.full(
                                (inputs.size(0), stride - inputs.size(1) + 1),
                                -100,
                                device=inputs.device,
                                dtype=inputs.dtype)

                            inputs = torch.cat((inputs, pad), dim=-1)
                            target_ids = torch.cat((target_ids, target_pad),
                                                   dim=-1)

                        inp.copy_(inputs)
                        g.replay()

                    else:
                        inp = input_ids[:, i:i + stride]
                        output = model(inp,
                                       use_cache=False,
                                       reset=i == 0,
                                       output_attentions=False)
                        logits = output.logits

                    targets = target_ids[:, i + 1:i + stride + 1]
                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    )

                    nll_total += _nll.sum()
                    count_total += (targets >= 0).sum().item()
                    pbar.set_description(
                        f"ppl: {(nll_total / count_total).exp()}")

                    print(f"\n\n{(nll_total / count_total).exp()=}\n\n")

                    l, u = j * args.batch_size, (j + 1) * args.batch_size
                    nll_individual[l:u] += _nll.cpu().sum(dim=-1)
                    count_individual[l:u] += (targets >= 0).sum(dim=-1).cpu()
                    step += stride
                    if step == 1024:
                        exit()

                    if step % 100 == 0:
                        # ppl = torch.exp(nll_total / count_total).item()
                        # run.track(ppl,
                        #           name="ppl-pg19",
                        #           step=step,
                        #           context={"subset": "test"})
                        # pbar.set_description(f"{ppl=:.6f}")

                        book_idx = l + torch.arange(args.batch_size)
                        book_ppl = torch.exp(nll_individual[book_idx] /
                                             count_individual[book_idx])

                        stats = {}
                        for k in range(args.batch_size):
                            key = f"ppl-pg19-book-{l + k}"
                            val = book_ppl[k].item()
                            # only track items which have not reached EOS
                            if targets.reshape(-1)[k] >= 0:
                                stats[key] = val

                        run.track(stats, step=i, context={"subset": "test"})

    ppl = torch.exp(nll_total / count_total).item()

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    if args.method == "sink":
        # use the aim tracker results for sink based methods
        pass
    else:
        with open(
                f'./cache/llama_eval/pg19-{args.method}-{args.model}-{args.comment}.json',
                'w') as f:
            json.dump({'ppl': ppl, "all_nll": all_nll}, f)

    # print(f'PPL: {ppl:.4f}')
