import os
import torch
import datasets
from torch.utils.data import Dataset


def cache_tokenized(dataset, tokenizer):
    os.makedirs('./cache/pg19', exist_ok=True)
    cache_path = './cache/pg19/tokenized.pth'
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    else:
        text = []
        for i in range(len(dataset)):
            print(f"loading book {i} from dataset")
            entry = dataset[i]
            text.append(entry['text'])

        print("tokenizing")
        ids = tokenizer(
            "\n\n".join(text),
            return_tensors='pt',
            truncation=False,
        ).input_ids

        print(f"{ids.size()=}")
        print("saving")
        torch.save(ids, cache_path)
        return cache_tokenized(dataset, tokenizer)


class PG19Streaming(Dataset):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset('emozilla/pg19-test')['test']

        self.inputs = cache_tokenized(self.dataset, self.tokenizer)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.inputs, self.inputs


if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'togethercomputer/LLaMA-2-7B-32K')
    ds = PG19Streaming(tokenizer)

    print(f"{len(ds)=}")
    print(f"{ds[0][0].size()=}")
