import os
import torch
import datasets
import warnings
from torch.utils.data import Dataset
import numpy as np


def cache_tokenized(dataset, tokenizer):
    print(tokenizer, vars(tokenizer))
    os.makedirs('./cache/pg19', exist_ok=True)

    name = tokenizer.name_or_path.replace("/", "-")
    cache_path = f'./cache/pg19/{name}-tokenized.pth'

    if os.path.exists(cache_path):
        print(f"loading cached tokenized books from: {cache_path}")
        return torch.load(cache_path)
    else:
        text = []
        for i in range(len(dataset)):
            print(f"loading book {i} from dataset")
            entry = dataset[i]
            text.append(entry['text'])

        print("tokenizing books")
        tokenized = [
            tokenizer(t, return_tensors="pt", truncation=False).input_ids
            for t in text
        ]

        for v in tokenized:
            print(v.size())

        print(f"{len(tokenized)=}")
        print("saving")
        torch.save(tokenized, cache_path)
        return cache_tokenized(dataset, tokenizer)


# def cache_tokenized(dataset, tokenizer):
#     print(tokenizer, vars(tokenizer))
#     os.makedirs('./cache/pg19', exist_ok=True)
#
#     name = tokenizer.name_or_path.replace("/", "-")
#     cache_path = f'./cache/pg19/{name}-tokenized.pth'
#
#     text = []
#     for i in range(len(dataset)):
#         entry = dataset[i]
#         print(f"{i} {entry['short_book_title']}")
#         # text.append(entry['text'])


class PG19Streaming(Dataset):

    def __init__(self, tokenizer, batch_size=24):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset('emozilla/pg19-test')['test']
        self.batch_size = batch_size

        self.inputs = cache_tokenized(self.dataset, self.tokenizer)

        lens = [v.size(1) for v in self.inputs]
        selected = np.argsort(lens)

        selected = selected[20:20 + 24].tolist() + selected[19:20].tolist()
        warnings.warn(
            f"there was a subset mismatch for qwen and llama tokenizers. Adding book 19:20 fixed it. If you are using a different tokenizer, you need to make sure you select the proper subset based on these indices. {selected} {len(selected)=}"
        )
        print(f"total books {np.argsort(lens)=}")
        print(f"selected book indices: {selected=}")

        # sort the books to pick a chunk of similarly sized books for a batch
        self.inputs = sorted(self.inputs, key=lambda x: x.size(1))
        sub = self.inputs[20:20 + 24] + self.inputs[19:20]

        print([v.size(1) for v in sub])
        print(torch.tensor([v.size(1) for v in sub]).cumsum(0))
        exit()
        self.inputs = sub

    def __len__(self):
        return (len(self.inputs) // self.batch_size)

    def __getitem__(self, idx) -> int:
        if idx >= len(self):
            raise IndexError("Index out of range")

        inputs = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        max_size = max([v.size(1) for v in inputs])

        out_inputs, out_labels = [], []
        for v in inputs:
            out_inputs.append(
                torch.cat((v,
                           torch.zeros(1,
                                       max_size - v.size(1),
                                       device=v.device,
                                       dtype=v.dtype)),
                          dim=-1))

            out_labels.append(
                torch.cat((v,
                           torch.full((1, max_size - v.size(1)),
                                      fill_value=-100,
                                      device=v.device,
                                      dtype=v.dtype)),
                          dim=-1))

        return torch.cat(out_inputs, dim=0), torch.cat(out_labels, dim=0)


if __name__ == '__main__':
    import transformers
    print("Llama")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'togethercomputer/LLaMA-2-7B-32K')
    ds = PG19Streaming(tokenizer)

    print(f"{len(ds)=}")
    for i, (x, y) in enumerate(ds):
        print(f"{i} {x.size()=} {y.size()=}")

    print("Qwen")
    tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen1.5-7B')
    ds = PG19Streaming(tokenizer)

    print("Qwen14b")
    tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen1.5-14B')
    ds = PG19Streaming(tokenizer)

    print("Llama3")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Meta-Llama-3-8B')
    ds = PG19Streaming(tokenizer)

    # print(f"{len(ds)=}")
    # for i, (x, y) in enumerate(ds):
    #     print(f"{i} {x.size()=} {y.size()=}")
