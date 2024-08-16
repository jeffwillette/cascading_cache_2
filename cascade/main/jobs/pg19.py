import os
import torch
from cascade.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import json
from aim import Run

from cascade.utils import MockRun
# import deepspeed
from cascade.models.llama.modeling_llama import LlamaDecoderLayer
from cascade.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from cascade.models.cascading_cache import CascadingKVCache


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
    stride, world_size = 1024, 1
    mdl = model.model

    # model = deepspeed.init_inference(
    #     model,
    #     tensor_parallel={"tp_size": args.world_size},
    #     replace_with_kernel_inject=False,
    #     dtype=args.infer_dtype,
    #     injection_policy=get_injection_policy(args.model),
    # )
    # mdl = model.module.model

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

    nll_total, count_total = 0, 0
    for j, (x, y) in enumerate(dataset):
        # print(f"\n\n\nwarning: truncating sequence for homogeneous heads rebuttal experiment\n\n\n")
        # x = x[:, :mdl.config.max_position_embeddings]
        # y = y[:, :mdl.config.max_position_embeddings]
        input_ids = x.cuda()
        target_ids = y.cuda()

        if past_key_values is not None:
            past_key_values.reset()

        print(
            f"starting batch of books: {input_ids.size()=} {target_ids.size()=}"
        )
        with tqdm(range(0, x.size(1) - 1, stride), ncols=150) as pbar:
            for i in pbar:
                with torch.no_grad():
                    # inp = input_ids[:, i:i + 1]
                    # use cache false means to use static cascading cache inside the model

                    inputs = input_ids[:, i:i + stride]
                    targets = target_ids[:, i + 1:i + stride + 1]
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
                        targets = torch.cat((targets, target_pad), dim=-1)

                    # I don't think this should happen because the dataset
                    # already pads these to be the same size for the batch.
                    # delete if this doesn't cause a problem
                    # elif targets.size(1) < inputs.size(1):
                    #     target_pad = torch.full(
                    #         (inputs.size(0), inputs.size(1) - targets.size(1)),
                    #         -100,
                    #         device=inputs.device,
                    #         dtype=inputs.dtype)

                    #     targets = torch.cat((targets, target_pad), dim=-1)
                    output = model(inputs, use_cache=use_cache, past_key_values=past_key_values)
                    past_key_values = output.past_key_values
                    logits = output.logits

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

                    l, u = j * args.batch_size, (j + 1) * args.batch_size
                    _nll = _nll.reshape(args.batch_size, -1)
                    nll_individual[l:u] += _nll.cpu().sum(dim=-1)
                    count_individual[l:u] += (targets >= 0).sum(dim=-1).cpu()
                    step += stride

                    book_idx = l + torch.arange(args.batch_size)
                    book_ppl = torch.exp(nll_individual[book_idx] / count_individual[book_idx])

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
