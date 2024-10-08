"""
cd cache/long_data_collection
wget https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/fine-tune/booksum.jsonl.zst
"""

import os
import json
import torch
import tqdm
import transformers


class BookSumDataset:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        json_path='./cache/long_data_collection/booksum.jsonl',
        max_seq_len=32768,
        truncation=True,
        for_eval=False,
        need_tokenization=False,
    ):

        with open(json_path, 'r') as f:
            lines = f.readlines()

        self.max_seq_len = max_seq_len
        self.truncation = truncation

        self.data = []
        for line in lines:
            # dict_keys(['text', 'prompt', 'completion'])
            self.data.append(json.loads(line))

        self.need_tokenization = need_tokenization
        if self.need_tokenization:
            self.tokenizer = tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.processed = []
            os.makedirs('./cache/long_data_collection', exist_ok=True)
            cache_path = f'./cache/long_data_collection/{tokenizer.name_or_path.split("/")[-1]}-booksum.pth'
            if not os.path.exists(cache_path):
                with tqdm.tqdm(self.data,
                               desc='tokenizing',
                               dynamic_ncols=True,
                               leave=False) as pbar:
                    for data in pbar:
                        assert tokenizer.eos_token is not None

                        text_ids = tokenizer(
                            data['text'] + ' ' + self.tokenizer.eos_token,
                            return_tensors='pt',
                            truncation=True,
                            max_length=self.max_seq_len,
                        )['input_ids'][0]
                        prompt_ids = tokenizer(
                            data['prompt'] + ' ' + self.tokenizer.eos_token,
                            return_tensors='pt',
                            truncation=True,
                            max_length=self.max_seq_len,
                        )['input_ids'][0]
                        completion_ids = tokenizer(
                            data['completion'] + ' ' +
                            self.tokenizer.eos_token,
                            return_tensors='pt',
                            truncation=True,
                            max_length=self.max_seq_len,
                        )['input_ids'][0]

                        input_ids = text_ids
                        target_ids = input_ids.clone()
                        target_ids[:-1] = input_ids[1:]
                        target_ids[-1] = -100
                        self.processed.append({
                            'input_ids': input_ids,
                            'labels': target_ids,
                            'text_ids': text_ids,
                            'prompt_ids': prompt_ids,
                            'completion_ids': completion_ids,
                        })
                        pbar.set_description(f't[{tuple(input_ids.shape)}]')
                torch.save(self.processed, cache_path)
            else:
                print('loading cache')
                self.processed = torch.load(cache_path)
                print('loaded', cache_path)
            print('loaded booksum', len(self.processed))

        self.for_eval = for_eval

    def __len__(self):
        if self.need_tokenization:
            return len(self.processed)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.need_tokenization:
            entry = self.processed[idx]
            if self.for_eval:
                return entry['prompt_ids'], entry['completion_ids']
            else:
                return entry['input_ids'], entry['labels']
        else:
            entry = self.data[idx]
            assert self.for_eval

            return entry['prompt'], entry['completion']


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/d1/dataset/llama/models/llama_v3.1/Meta-Llama-3.1-8B-Instruct"
    )
    ds = BookSumDataset(
        tokenizer,
        max_seq_len=None,
        truncation=False,
        need_tokenization=False,
        for_eval=True
    )

    test_fname = "saves/llama_eval/booksum/test_idx.pt"
    train_fname = "saves/llama_eval/booksum/train_idx.pt"
    if not os.path.exists(test_fname):
        train_idx, test_idx = train_test_split(list(range(len(ds))), test_size=0.05)
        train_idx, test_idx = torch.tensor(train_idx), torch.tensor(test_idx)
        torch.save(train_idx, train_fname)
        torch.save(test_idx, test_fname)
    else:
        train_idx = torch.load(train_fname).tolist()
        test_idx = torch.load(test_fname).tolist()

    ds = Subset(ds, test_idx)

    total_tokens, completions = 0, 0
    max_len = 0
    for idx in tqdm.tqdm(range(len(ds))):
        prompt, completion = ds[idx]

        prompt = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0]
        completion = tokenizer(completion, return_tensors="pt", truncation=False).input_ids[0]
        token_length = prompt.shape[0]
        compl_length = completion.shape[0]
        total_tokens += token_length
        completions += compl_length

        max_len = max(max_len, token_length + compl_length)

        print(f"{idx=} {token_length=} {compl_length=}")
    print(f"{total_tokens=} {completions=}")
    print(f"{max_len=}")
