import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse, json
from transformers import TextStreamer

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench


def job_ppl(args, model, tokenizer, device):
    os.makedirs('./cache', exist_ok=True)
    cache_path = './cache/llama_eval.pth'
    if not os.path.exists(cache_path):
        # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        # test = load_dataset("openwebtext", split="train")
        print(test)
        encodings = tokenizer("\n\n".join(test["text"]),
                              return_tensors="pt").input_ids
        torch.save(encodings, cache_path)
    else:
        encodings = torch.load(cache_path)

    max_length = model.config.max_position_embeddings
    max_length = stride = args.stride if args.stride > 0 else model.config.max_position_embeddings
    # if model.model.config._umbc:
    #     max_length = stride = max_length // 2
    seq_len = encodings.size(1)

    nlls = []
    prev_end_loc = 0
    with tqdm(range(0, seq_len, stride)[:args.count]) as pbar:
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            if args.method != "umbc":
                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood.cpu())

            elif args.method == "umbc":
                with torch.no_grad():
                    model.model.model.sse.post_forward_mbc_cleanup()
                    output_logits = torch.zeros(input_ids.size(0),
                                                input_ids.size(1),
                                                32000,
                                                device=input_ids.device)

                    past_key_values = None
                    for i in tqdm(range(target_ids.size(1))):
                        inputs = input_ids[:, i:i + 1]

                        output = model(
                            inputs,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )
                        output_logits[:, i] = output.logits[:, 0]
                        past_key_values = output.past_key_values

                        if i % 100 == 0:
                            ce = torch.nn.functional.cross_entropy(
                                output_logits[:, :-1].reshape(
                                    -1, output_logits.size(-1)),
                                target_ids[:, 1:output_logits.size(1) +
                                           1].view(-1),
                            )
                            print(f"intermediate ce: {ce.item()}")

                    logits = output_logits[:, :-1].reshape(
                        -1, output_logits.size(-1))
                    t = target_ids[:, 1:].reshape(-1)
                    loss = torch.nn.functional.cross_entropy(logits, t)
                    nlls.append(loss.cpu())
                    model.model.model.sse.post_forward_mbc_cleanup()
            elif args.method == "umbc-full-eval":
                model.model.sse.post_forward_mbc_cleanup()

                with torch.no_grad():
                    outputs = model(input_ids,
                                    labels=target_ids,
                                    use_cache=False)
                    neg_log_likelihood = outputs.loss

                nlls.append(neg_log_likelihood.cpu())
                print(torch.exp(neg_log_likelihood.cpu()).item())

            prev_end_loc = end_loc
            ppl = torch.exp(torch.stack(nlls).mean()).item()
            pbar.set_description(f"ppl: {ppl:.3f}")

            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean()).item()

    os.makedirs('./cache/llama_eval/', exist_ok=True)
    with open(f'./cache/llama_eval/ppl-{args.method}.json', 'w') as f:
        json.dump({'ppl': ppl}, f)

    print(f'PPL: {ppl:.4f}')
