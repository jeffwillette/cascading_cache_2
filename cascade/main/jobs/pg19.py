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
from aim import Run

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


def job_ppl_pg19(args, model, tokenizer, device):
    output_attentions = False
    if "attention-matrix-plot" in args.comment:
        model.model.setup_caches(args.world_size)
        model = model.to(args.infer_dtype).cuda()
        output_attentions = True
    elif args.method == "hyper":
        model.model.setup_hyper_attention()

        # model = model.to(args.infer_dtype).cuda()

        # only for Llama7B 3/4 layers which had precision errors with float16
        model = model.to(torch.bfloat16).cuda()

        # model = deepspeed.init_inference(
        #     model,
        #     tensor_parallel={"tp_size": args.world_size},
        #     replace_with_kernel_inject=False,
        #     dtype=args.infer_dtype,
        #     injection_policy=get_injection_policy(args.model),
        # )

    elif args.world_size == 1:
        model.model.setup_caches(args.world_size)
        model = model.to(args.infer_dtype).cuda()
        # model = torch.compile(model, mode="max-autotune", fullgraph=False)
    else:
        model.model.setup_caches(args.world_size)
        model = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": args.world_size},
            replace_with_kernel_inject=False,
            dtype=args.infer_dtype,
            injection_policy=get_injection_policy(args.model),
        )

    run = MockRun()
    if args.local_rank == 0:
        run = Run(experiment=f"{args.method}-pg19"
                  ) if not args.dev_run else MockRun()
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

    for j, (x, y) in enumerate(dataset):
        input_ids = x.cuda()
        target_ids = y.cuda()

        ############################ TEMPORARY HACK TO MATCH THE SUBSETS OF QWEN AND LLAMA
        # if j < 24:
        #     print(f"skipping book {j}")
        #     continue
        ############################ END HACK, COMMENT OUT WHEN DONE

        if args.method == "sink":
            print(
                f"starting batch of books: {input_ids.size()=} {target_ids.size()=}"
            )

            ATTN_ROWS_LIMIT = 8192
            attn_rows = []
            with tqdm(range(x.size(1) - 1), ncols=150) as pbar:
                for i in pbar:
                    with torch.no_grad():
                        inp = input_ids[:, i:i + 1]
                        # use cache false means to use static cascading cache inside the model
                        output = model(inp,
                                       use_cache=False,
                                       reset=i == 0,
                                       output_attentions=output_attentions)

                        if output_attentions:
                            attn_rows.append([
                                a.cpu().amax(dim=1).to(torch.float8_e4m3fn)
                                for a in output.attentions
                            ])
                            print(f"{len(attn_rows)=}")
                            if i == ATTN_ROWS_LIMIT - 1:
                                torch.save(
                                    attn_rows,
                                    f"./attention_visualization/book-{j}-window-{args.window}-cascades-{args.cascades}.pt"
                                )
                                break

                        logits = output.logits[:, -1:]
                        targets = target_ids[:, i + 1:i + 2]
                        _nll = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            targets.reshape(-1),
                            ignore_index=-100,
                            reduction="none",
                        )

                        nll_total += _nll.sum()
                        count_total += (targets >= 0).sum().item()

                        l, u = j * args.batch_size, (j + 1) * args.batch_size
                        nll_individual[l:u] += _nll.cpu()
                        count_individual[l:u] += (targets
                                                  >= 0).sum(dim=-1).cpu()
                        step += 1

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
                                if targets.view(-1)[k] >= 0:
                                    stats[key] = val

                            run.track(stats,
                                      step=i,
                                      context={"subset": "test"})
        elif args.method == "hyper":
            with torch.no_grad():
                max_len = 32768
                input_ids = input_ids[:, :model.config.max_position_embeddings]
                target_ids = target_ids[:, 1:model.config.
                                        max_position_embeddings + 1]

                print(input_ids.size())
                print(target_ids.size())
                if target_ids.size(-1) + 1 == input_ids.size(-1):
                    input_ids = input_ids[:, :-1]

                # input_ids = input_ids[:, :4096]
                # target_ids = target_ids[:, 1:4096 + 1]
                # print(f"{input_ids=} {target_ids=}")

                attn_out = model(input_ids, use_cache=False)

                logits = attn_out.logits
                print(f"{logits.size()=}")

                logit_lst = logits.split(100, dim=1)
                target_lst = target_ids.split(100, dim=1)

                nll_cum = 0
                count = 0
                for l, t in zip(logit_lst, target_lst):
                    nll = torch.nn.functional.cross_entropy(l.reshape(
                        -1, l.size(-1)).float(),
                                                            t.reshape(-1),
                                                            ignore_index=-100,
                                                            reduction="sum")
                    nll_cum += nll
                    count += t.size(1)

                nll = nll_cum / count

                all_nll.append(nll.item())
                print(f"{all_nll=}")
                count_total += 1
                nll_total += nll
                print(f"book: {j=} perplexity: {torch.exp(nll).item()=}")

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
