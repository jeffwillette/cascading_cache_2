import json
import os
from pyrouge import Rouge155

import pathlib

import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

from cascade.dataset.booksum import BookSumDataset
from cascade.models.cascading_cache import CascadingKVCache
from torch.utils.data import Subset
import math
import numpy as np

import subprocess
import logging


class StopAfterStringIsGenerated(LogitsProcessor):
    def __init__(self, base_len: int, tokenizer):
        super().__init__()

        self.base_len = base_len
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.base_len:
            decoded = self.tokenizer.batch_decode(input_ids[:, self.base_len:])
            ends_with_answer = torch.tensor([s.endswith("</s>") for s in decoded], device=scores.device)
            forced_eos = torch.full((scores.size(1),), -float("inf"), device=scores.device)
            forced_eos[self.tokenizer.eos_token_id] = 0

            # Force generation of EOS after a space
            scores[ends_with_answer] = forced_eos
        return scores


PROMPT_FIRST_ONLY = os.getenv('PROMPT_FIRST_ONLY', '1') == '1'


def generate_summary(args, model, tokenizer, device, idx, item, out_dir):
    inputs, completion = item

    if (out_dir / f"out_{idx}.txt").exists() and not args.overwrite:
        with open(out_dir / f"out_{idx}.txt", 'r') as f:
            return f.read()

    tokenizer.truncation_side = 'left'

    assert hasattr(model, 'config')
    assert hasattr(model.config, 'max_position_embeddings')

    messages = [
        {"role": "system", "content": "You are a helpful chat bot that summarizes books."},
        {"role": "user", "content": f"Summarize the following text in about 300 words:\n{inputs}"},
    ]

    if messages[1]["content"].endswith('</s>'):
        messages[1]["content"] = messages[1]["content"][:-4]

    max_length = model.config.max_position_embeddings - args.max_tokens
    truncation = True
    if args.method != "vanilla":
        max_length = None
        truncation = False

    inputs = tokenizer.apply_chat_template(messages,
                                           return_tensors='pt',
                                           max_length=max_length,
                                           truncation=truncation,
                                           )

    seq_len = inputs.shape[-1]
    print(f"seq_len: {seq_len}")

    additional_args = {}
    args.no_sample = True
    if not args.no_sample:
        additional_args = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    if args.method == "sink":
        max_seq_len = int(2 ** math.floor(np.log2(seq_len / 2)))
        # max_seq_len = args.window
        max_seq_len = min(max_seq_len, 16384)
        print(f"{max_seq_len=}")
        window = max_seq_len // args.cascades

        mdl = model.model
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

    output = model.generate(
        inputs=inputs.cuda(),
        past_key_values=past_key_values,
        attention_mask=torch.ones((1, inputs.shape[-1]), dtype=torch.long, device='cuda'),
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=LogitsProcessorList([
            StopAfterStringIsGenerated(inputs.shape[-1], tokenizer)
        ]),
        **additional_args,
    )
    output: str = tokenizer.decode(
        output[0][seq_len:].data.cpu(),
        skip_special_tokens=True,
    )
    if output.endswith('</s>'):
        output = output[:-4]
    output = output.strip()

    return output


def install_rogue():
    logger = logging.getLogger()

    ROUGE_HOME = os.environ.get('ROUGE_HOME', "cache/ROUGE-1.5.5")
    if "ROUGE_HOME" not in os.environ:
        logger.info("ROUGE_HOME not set, using default location %s", ROUGE_HOME)

    if not os.path.exists(ROUGE_HOME):
        logger.info("ROUGE_HOME=%s not a directory.", ROUGE_HOME)
        try:
            logger.info("Installing rouge Perl script to {ROUGE_HOME} - this will take a few seconds")
            subprocess.run(
                ["curl", "-L", "https://github.com/Yale-LILY/SummEval/tarball/7e4330d", "-o", "project.tar.gz", "-s"])
            subprocess.run(["tar", "-xzf", "project.tar.gz"])
            subprocess.run(["mv", "Yale-LILY-SummEval-7e4330d/evaluation/summ_eval/ROUGE-1.5.5/", ROUGE_HOME])
            subprocess.run(["rm", "project.tar.gz"])
            subprocess.run(["rm", "-rf", "Yale-LILY-SummEval-7e4330d/"])
        except subprocess.CalledProcessError as err:
            logger.error(
                "Failed to install the rouge Perl script; please install manually and set the ROUGE_HOME environment variable.")
            raise err

    return ROUGE_HOME


def generate_samples(args, model, tokenizer, device, out_dir):
    is_vllm = False
    if is_vllm:
        # we do not access to tokenizer.
        tokenizer = None

    dataset = BookSumDataset(
        tokenizer=tokenizer,
        for_eval=True,
        need_tokenization=False,
    )

    test_fname = "saves/llama_eval/booksum/test_idx.pt"
    train_fname = "saves/llama_eval/booksum/train_idx.pt"
    if not os.path.exists(test_fname):
        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.05)
        train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)
        torch.save(train_idx, train_fname)
        torch.save(test_idx, test_fname)
    else:
        train_idx = torch.load(train_fname).tolist()
        test_idx = torch.load(test_fname).tolist()

    test_dataset = Subset(dataset, test_idx)

    outputs = []

    for idx, item in enumerate(tqdm(test_dataset, dynamic_ncols=True, leave=True, desc="booksum")):
        inputs, completion = item

        if is_vllm:
            raise NotImplementedError("refer to original job file: https://github.com/gmlwns2000/hip-attention/blob/main/hip/main/jobs/booksum.py")
        else:
            output = generate_summary(args, model, tokenizer, device, idx, item, out_dir)

        output_summary = output.replace('\n', '\\n')[:200]
        tqdm.write(f"[{idx:<7}] Summary: {output_summary}[...]")
        with open(out_dir / f"out_{idx}.txt", 'w') as f:
            f.write(output)
        outputs.append(output)

        fname = f"saves/llama_eval/booksum/reference/ref_{idx}.txt"
        if not os.path.exists(fname):
            with open(fname, 'w') as f:
                if isinstance(completion, str):
                    f.write(completion)
                else:
                    f.write(tokenizer.decode(completion, skip_special_tokens=True))


MAX_NEW_TOKENS = 256


def evaluate_rouge(args, model, tokenizer, device, out_dir: pathlib.Path):
    for node in out_dir.glob('*'):
        if node.is_file():
            content = node.read_text()
            ids = tokenizer(content, truncation=True, max_length=256).input_ids
            content = tokenizer.decode(ids, skip_special_tokens=True)
            node.write_text(content)

    rouge_dir = install_rogue()

    r = Rouge155(rouge_dir=rouge_dir)
    r.system_dir = out_dir  # "system" is the one we want to measure
    r.model_dir = "saves/llama_eval/booksum/reference"  # "model" is the gold standard (i.e. human summaries)
    r.system_filename_pattern = r'out_(\d+)\.txt'
    r.model_filename_pattern = r'ref_#ID#\.txt'

    output = r.convert_and_evaluate()
    print("R: Recall, P: Precision, F: F1 score")
    print(output)
    output_dict = r.output_to_dict(output)
    with open(out_dir / "rouge_scores.json", 'w') as f:
        json.dump(output_dict, f, indent=2)


@torch.no_grad()
def job_booksum(args, model, tokenizer, device):
    args.overwrite = False

    f = f'saves/llama_eval/booksum/{args.method}-{args.model}-{args.comment}-window-{args.window}-' + \
        f'cascades-{args.cascades}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}-' + \
        f'cascade-stride-{args.cascade_stride}-homogeneous-heads-{args.homogeneous_heads}-comment-{args.comment}.json'

    out_dir = pathlib.Path(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("saves/llama_eval/booksum/reference").mkdir(parents=True, exist_ok=True)

    # generate_samples(args, model, tokenizer, device, out_dir)
    evaluate_rouge(args, model, tokenizer, device, out_dir)
    print(args)
