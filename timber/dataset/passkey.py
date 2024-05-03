import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np

PREFIX = "There is important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the the important information. "
FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
QUERY = "What is the pass key? The pass key is "


def interpolate_passkey(k):
    return f"The pass key is {k}. Remember it. {k} is the pass key. "


def gen_text():
    os.makedirs('./cache/passkey', exist_ok=True)
    text_path = './cache/passkey/passkey.txt'
    label_path = './cache/passkey/passkey-label.txt'

    if os.path.exists(text_path):
        print(f"loading cached text from: {text_path}")

        with open(text_path) as f:
            out = f.readlines()
        out = [o[:-1] for o in out]

        with open(label_path) as f:
            out_labels = f.readlines()
        out_labels = [o[:-1] for o in out_labels]

        return out, out_labels
    else:
        print("generating text")
        prefix_len = len(PREFIX[:-1].split(" "))
        filler_len = len(FILLER_TEXT[:-1].split(" "))
        query_len = len(QUERY[:-1].split(" "))

        inputs, targets = [], []
        prompt_lens = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
        for l in prompt_lens:
            n_fillers = (l - prefix_len - query_len) // filler_len + 1
            for i in range(50):

                text = [PREFIX] + [FILLER_TEXT] * n_fillers

                k = np.random.randint(10000, 50000)

                key_phrase = interpolate_passkey(k)
                target = f"{k}"

                insert_loc = np.random.randint(2, len(text) - 1)
                text = text[:insert_loc] + \
                    [key_phrase] + text[insert_loc:] + [QUERY]

                text = "".join(text)

                inputs.append(text)
                targets.append(target)

        with open(text_path, "w+") as f:
            for t in inputs:
                f.write(t)
                f.write("\n")

        with open(label_path, "w+") as f:
            for t in targets:
                f.write(t)
                f.write("\n")

        return gen_text()


def gen_dataset(tokenizer):
    print(tokenizer, vars(tokenizer))

    os.makedirs('./cache/passkey', exist_ok=True)
    name = tokenizer.name_or_path.replace("/", "-")
    if name[0] == "-":
        name = name[1:]

    cache_path = f'./cache/passkey/{name}-tokenized.pth'

    if os.path.exists(cache_path):
        print(f"loading cached tokenized text from: {cache_path}")
        return torch.load(cache_path)
    else:
        inputs, targets = gen_text()

        x, y = [], []
        for inp, tgt in zip(inputs, targets):
            x += [
                tokenizer(inp, return_tensors="pt", truncation=False).input_ids
            ]
            y += [
                tokenizer(
                    tgt,
                    return_tensors="pt",
                    truncation=False,
                    add_special_tokens=False,
                ).input_ids
            ]

        for i in range(len(x)):
            print(x[i].size(), y[i].size())

        torch.save([x, y], cache_path)
        return gen_dataset(tokenizer)


class Passkey(Dataset):

    def __init__(self, tokenizer, batch_size=10):
        self.tokenizer = tokenizer
        self.dataset = gen_dataset(self.tokenizer)
        self.batch_size = batch_size

        self.inputs = self.dataset[0]
        self.targets = self.dataset[1]

    def __len__(self):
        return (len(self.inputs) // self.batch_size)

    def __getitem__(self, idx) -> int:
        if idx >= len(self):
            raise IndexError("Index out of range")

        inputs = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        targets = self.targets[idx * self.batch_size:(idx + 1) *
                               self.batch_size]

        # max_size = max([v.size(1) for v in inputs])
        # max_size_target = max([v.size(1) for v in targets])

        # out_inputs, out_targets = [], []
        # for v, u in zip(inputs, targets):
        #     out_inputs.append(
        #         torch.cat((v, -100 * torch.ones(
        #             1, max_size - v.size(1), device=v.device, dtype=v.dtype)),
        #                   dim=-1))

        #     out_targets.append(
        #         torch.cat((u,
        #                    torch.full((1, max_size_target - u.size(1)),
        #                               fill_value=-100,
        #                               device=u.device,
        #                               dtype=u.dtype)),
        #                   dim=-1))

        return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'togethercomputer/LLaMA-2-7B-32K')
    ds = Passkey(tokenizer)

    print(f"{len(ds)=}")
    for i, (x, y) in enumerate(ds):
        print(f"{i} {x.size()=} {y.size()=}")
