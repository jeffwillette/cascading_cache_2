import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np

# PREFIX = "There is important info hidden inside a lot of irrelevant text. Find the important info and memorize it. I will quiz you about the the important information. "
# PREFIX = "There is a pass key token [KEY]<numbers>[/KEY] hidden inside a lot of irrelevant text. Find the <numbers> in the [KEY] token and memorize them. I will quiz you about the the <numbers>. Do not forget the <numbers> inside the [KEY]"
# FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again"
# FILLER_TEXT = "If you see <numbers> inside a [KEY], don't forget the <numbers> inside the [KEY]<numbers>[/KEY]."
# QUERY = "What is the pass key [KEY]<numbers>[/KEY]? The pass key <numbers> are "

PREFIX = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

There is a pass key hidden inside a lot of irrelevant text. Find the pass key and memorize it. I will quiz you about the the pass key.

"""
# FILLER_TEXT = """The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. """
QUERY = """\n So now, I will ask the question. What is the five digit pass key?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Sure! I surely remember the five digits pass key. The pass key is $"""


def interpolate_passkey(k):
    keyline = f"HERE IS THE PASSKEY! The pass key is {k}. {k} is the pass key. **the pass key is {k}** LOOK BEHIND FOR PASSKEY"
    return f"=== NOW IMPORTANT INFORMATION STARTS ===\n{keyline}\nREPEAT THE INFORMATION\n{keyline}\n=== IMPORTANT INFORMATION STOPS ==="


def gen_filler_text(num_words):
    import random

    out = []
    word_file = "/usr/share/dict/words"
    WORDS = open(word_file).read().splitlines()
    for i in range(num_words):
        out += [random.choice(WORDS)]

    return " ".join(out)


FILLER_TEXT = gen_filler_text(128000)

print(FILLER_TEXT[:10000])


def gen_text(tokenizer):
    prefix_tok = tokenizer(PREFIX, return_tensors="pt", truncation=False).input_ids[0]
    prefix_len = prefix_tok.shape[0]

    query_tok = tokenizer(QUERY, return_tensors="pt", truncation=False).input_ids[0]
    query_len = query_tok.shape[0]

    filler_tok = tokenizer(FILLER_TEXT, return_tensors="pt", truncation=False).input_ids[0]
    # filler_tok = torch.randperm(128000)

    inputs, targets, len_loc = [], [], []
    prompt_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    insert_locs = [0.2, 0.4, 0.6, 0.8, 1.0]

    for l in prompt_lens:
        n_fillers = (l - prefix_len - query_len)
        filler = filler_tok[:n_fillers]
        for loc in insert_locs:
            for i in range(25):

                k = np.random.randint(10000, 100000)

                key_phrase = interpolate_passkey(k)
                target = f"{k}"
                key_tok = tokenizer(key_phrase, return_tensors="pt", truncation=False).input_ids[0]
                local_filler = filler[:-key_tok.shape[0]]

                start, end = int((loc - 0.2) * l), int(loc * l)
                insert_loc = np.random.randint(start, end)
                tokens = torch.cat((prefix_tok, local_filler[:insert_loc], key_tok, local_filler[insert_loc:], query_tok))

                inputs.append(tokens)
                targets.append(target)
                len_loc.append((l, loc))

    return inputs, targets, len_loc


class Passkey(Dataset):

    def __init__(self, tokenizer, batch_size=10):
        self.tokenizer = tokenizer
        inputs, targets, len_loc = gen_text(self.tokenizer)
        self.batch_size = batch_size

        self.inputs = inputs
        self.targets = targets
        self.len_loc = len_loc

    def __len__(self):
        return (len(self.inputs) // self.batch_size)

    def __getitem__(self, idx) -> int:
        if idx >= len(self):
            raise IndexError("Index out of range")

        inputs = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        len_loc = self.len_loc[idx * self.batch_size:(idx + 1) * self.batch_size]

        return torch.stack(inputs, dim=0), targets, len_loc


if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct')

    token_prefix = tokenizer(PREFIX, return_tensors="pt",
                             truncation=False).input_ids
    print(f"{token_prefix.size()=}")
    ds = Passkey(tokenizer)

    print(f"{len(ds)=}")
    for i, (x, y) in enumerate(ds):
        print(f"{i} {x.size()=} {y.size()=}")
