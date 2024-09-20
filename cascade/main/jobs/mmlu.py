import json
import os
from cascade.models.cascading_cache import CascadingKVCache
import time
import math
import torch
import transformers
from datasets import load_dataset
import tqdm
import numpy as np
# import deepspeed

from cascade.utils import seed
# from cascade.main.jobs.pg19 import get_injection_policy

MMLU_FORMAT = """{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

MMLU_FORMAT_TARGET_QUESTION = """I want you to answer the following multiple choice question about {subject_name}.

{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:{answer_placeholder}"""


MMLU_SUBJECTS = [
    'business_ethics',
    'clinical_knowledge',
    'medical_genetics',
    'high_school_us_history',
    'high_school_physics',
    'high_school_world_history',
    'virology',
    'high_school_microeconomics',
    'econometrics',
    'college_computer_science',
    'high_school_biology',
    'abstract_algebra',
    'professional_accounting',
    'philosophy',
    'professional_medicine',
    'nutrition',
    'global_facts',
    'machine_learning',
    'security_studies',
    'public_relations',
    'professional_psychology',
    'prehistory',
    'anatomy',
    'human_sexuality',
    'college_medicine',
    'high_school_government_and_politics',
    'college_chemistry',
    'logical_fallacies',
    'high_school_geography',
    'high_school_european_history',  # 9600
    'elementary_mathematics',
    'human_aging',
    'college_mathematics',
    'high_school_psychology',
    'formal_logic',
    'high_school_statistics',
    'international_law',
    'high_school_mathematics',
    'high_school_computer_science',
    'conceptual_physics',
    'miscellaneous',
    'high_school_chemistry',
    'marketing',
    'professional_law',
    'management',
    'college_physics',
    'jurisprudence',
    'world_religions',
    'sociology',
    'us_foreign_policy',
    'high_school_macroeconomics',
    'computer_security',
    'moral_scenarios',
    'moral_disputes',
    'electrical_engineering',
    'astronomy',
    'college_biology'
]


def format_mmlu(question,
                number,
                subject_name,
                target_question=False,
                answer_placeholder=""):
    """
    {'input': 'A "dished face" profile is often associated with',
    'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.',
    'B': 'a recessive maxilla due to failure of elongation of the cranial base.',
    'C': 'an enlarged frontal bone due to hydrocephaly.',
    'D': 'defective development of the maxillary air sinus.',
    'target': 'B'}
    """

    if not target_question:
        return MMLU_FORMAT.format(
            subject_name=subject_name,
            number=number,
            question=question['input'],
            choice_a=question['A'],
            choice_b=question['B'],
            choice_c=question['C'],
            choice_d=question['D'],
        )
    return MMLU_FORMAT_TARGET_QUESTION.format(
        subject_name=subject_name,
        number=number,
        question=question['input'],
        choice_a=question['A'],
        choice_b=question['B'],
        choice_c=question['C'],
        choice_d=question['D'],
        answer_placeholder=answer_placeholder,
    )


def exam_mmlu(model, tokenizer: transformers.PreTrainedTokenizer, texts, args, subject_stats, subject_name):

    avg_len = subject_stats[subject_name][0]
    target_window = 2**int(np.log2(avg_len))
    target_window = min(2048, target_window)

    def gather_token_ids(candidates):
        ids = []
        for cand in candidates:
            ids.append(tokenizer(cand).input_ids[-1])
        return ids

    tokens_a = gather_token_ids(
        ['A', ':A', ': A', ':  A', 'a', ':a', ': a', ':  a'])
    tokens_b = gather_token_ids(
        ['B', ':B', ': B', ':  B', 'b', ':b', ': b', ':  b'])
    tokens_c = gather_token_ids(
        ['C', ':C', ': C', ':  C', 'c', ':c', ': c', ':  c'])
    tokens_d = gather_token_ids(
        ['D', ':D', ': D', ':  D', 'd', ':d', ': d', ':  d'])

    tokenizer.truncation_side = 'left'

    assert hasattr(model, 'config')
    assert hasattr(model.config, 'max_position_embeddings')

    text_len = len(texts)
    if text_len < args.batch_size:
        texts += ["null text"] * (args.batch_size - text_len)

    max_length = model.config.max_position_embeddings
    if args.method == "vanilla" and "truncate" in args.comment:
        max_length = target_window

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        max_length=max_length,
        truncation=False if args.method == "sink" else True,
        padding=True,
    )

    use_cache, past_key_values = False, None
    if args.method == "sink":
        # max_seq_len = int(2 ** math.floor(np.log2(inputs.input_ids.size(1) / 2)))
        # max_seq_len = min(max_seq_len, 16384)
        max_seq_len = target_window
        window = max_seq_len // args.cascades
        mdl = model.model

        use_cache = True
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
            verbose=False,
        )

    # inputs["input_ids"] = inputs["input_ids"][:, :50]
    # inputs["attention_mask"] = inputs["attention_mask"][:, :50]

    seq_lens = inputs.attention_mask[:text_len].sum(dim=-1).tolist()

    with torch.no_grad():
        if args.method == "sink":
            past_key_values.reset(verbose=False)
            input_ids = inputs["input_ids"].cuda()

            # these are the first outputs after the last of the prompt sequence
            pred_indices = inputs.attention_mask[:text_len].sum(
                dim=-1, keepdim=True) - 1

            # print(f"{pred_indices=}")

            logits = []
            for i in range(0, input_ids.size(1), args.cascade_stride):
                output = model(
                    input_ids[:, i: i + args.cascade_stride],
                    use_cache=use_cache,
                    past_key_values=past_key_values
                )
                logits += [output.logits[:text_len].cpu().half()]
                past_key_values = output.past_key_values

            logits = torch.cat(logits, dim=1).to(pred_indices.device)

            pred_indices = pred_indices.unsqueeze(-1).repeat(
                1, 1, logits.size(-1))

            logits = torch.gather(logits.cpu(), 1, pred_indices)
            output = logits.softmax(dim=-1)

        else:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            output = model(**inputs).logits
            output = torch.softmax(output[:text_len], dim=-1)

    out_probs = []
    for i in range(output.size(0)):
        prob_a = max([output[i, -1, token].item() for token in tokens_a])
        prob_b = max([output[i, -1, token].item() for token in tokens_b])
        prob_c = max([output[i, -1, token].item() for token in tokens_c])
        prob_d = max([output[i, -1, token].item() for token in tokens_d])
        probs = [('A', prob_a), ('B', prob_b), ('C', prob_c), ('D', prob_d)]

        probs = list(sorted(probs, key=lambda x: x[1], reverse=True))
        out_probs.append(probs)

    return out_probs, seq_lens


def get_fewshots(dataset, subject_name, shift=1):
    few_shots = []
    for question in dataset['train']:
        text = format_mmlu(question, len(few_shots) + shift, subject_name)
        choice = question['target']
        text += choice
        few_shots.append(text)
    for question in dataset['validation']:
        text = format_mmlu(question, len(few_shots) + shift, subject_name)
        choice = question['target']
        text += choice
        few_shots.append(text)
    few_shots = few_shots[:20]
    return few_shots


def format_mmlu_plain(question, subject_name, few_shots, tokenizer):
    text = format_mmlu(
        question,
        len(few_shots) + 1,
        subject_name,
        target_question=True,
        answer_placeholder="",
    )

    # fs_q = [v[:-1] for v in few_shots]
    # fs_a = [v[-1:] for v in few_shots]

    # messages = [
    #     {"role": "system", "content": f"The following are multiple choice questions (with answers) about {subject_name}"},
    # ]

    # for q, a in zip(fs_q, fs_a):
    #     messages += [{"role": "user", "content": q}, {"role": "system", "content": a}]
    # messages += [{"role": "user", "content": text}]

    # text = tokenizer.apply_chat_template(messages, tokenize=False)
    # text += "<|start_header_id|>system<|end_header_id|>\n"

    questions = "\n\n".join(few_shots)
    text = "The following are multiple choice questions (with answers) " + \
        f"about {subject_name}:\n\n{questions}\n\n" + \
        text

    truth = question['target']
    return text, truth


def evaluate_mmlu(args, model, tokenizer, subject_name, json_path, subject_stats):
    dataset = load_dataset('lukaemon/mmlu', subject_name, trust_remote_code=True)

    few_shots = get_fewshots(dataset, subject_name, shift=1)

    t_start = time.time()
    results = []
    n_correct = 0
    seq_len_sum = 0

    qanda = [format_mmlu_plain(question, subject_name, few_shots, tokenizer) for question in dataset["test"]]

    print("total len of questions and answers: ", len(qanda))
    n = math.ceil(len(qanda) / args.batch_size)

    with tqdm.tqdm(range(n), dynamic_ncols=True, leave=True,
                   desc=subject_name) as pbar:
        for i in pbar:
            input_slice = qanda[i * args.batch_size:(i + 1) * args.batch_size]

            texts = [v[0] for v in input_slice]
            truths = [v[1] for v in input_slice]

            batch_estimations, seq_lens = exam_mmlu(model, tokenizer, texts, args, subject_stats, subject_name)
            # print(f"{truths=} {batch_estimations=}")

            for estimations, truth, seq_len in zip(batch_estimations, truths,
                                                   seq_lens):
                estimation = estimations[0][0]
                correct = truth == estimation

                # print(truth, estimations, seq_len)
                if correct:
                    n_correct += 1
                seq_len_sum += seq_len

                results.append({
                    # 'text': text,
                    'truth': truth,
                    'estimations': estimations,
                    'estimation': estimation,
                    'correct': correct,
                    'seq_len': seq_len,
                })

            pbar.set_description(f"acc: {n_correct / len(results)}")

    elapsed = time.time() - t_start
    accuracy = (n_correct / len(results)) * 100
    avg_seq_len = seq_len_sum / len(results)
    print(f'{subject_name} = Accuracy: {accuracy:.4f} %, avg_seq_len: {avg_seq_len:.2f}. elapsed: {elapsed:.1f} s')

    with open(json_path, 'w') as f:
        json.dump(
            {
                'accuracy': accuracy,
                'avg_seq_len': avg_seq_len,
                'elapsed': elapsed,
                'model': args.model,
                'results': results,
            },
            f,
            indent=2)
        print('dumped', json_path)

    return accuracy


def job_mmlu(args, model, tokenizer, device):
    seed()

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()
    with open("./saves/llama_eval/mmlu-stats.json", "r") as f:
        subject_stats = json.load(f)

    accuracies = []
    for subject in MMLU_SUBJECTS:
        os.makedirs('./saves/llama_eval/mmlu/', exist_ok=True)
        json_path = f'./saves/llama_eval/mmlu/{subject}_{args.model}_{args.method}_comment_{args.comment}.json'
        if args.method == 'sink':
            json_path = f'./saves/llama_eval/mmlu/{subject}_{args.model}_{args.method}_window_{args.window}_' + \
                f'head_reduction_{args.head_reduction}_cascade_{args.cascades}_sinks_{args.sinks}_' + \
                f'homogeneous_heads_{args.homogeneous_heads}_cascade_stride_{args.cascade_stride}_comment_{args.comment}.json'

        if os.path.exists(json_path):
            print(f"skipping {subject} because the file already exists.")
            continue

        print(f"mean len for {subject}: {subject_stats[subject][0]=}")
        # if subject_stats[subject][0] < args.window:
        #     continue
        acc = evaluate_mmlu(args, model, tokenizer, subject, json_path, subject_stats)
        accuracies.append(acc)

    accuracy = np.array(accuracies).mean()
    print(f'MMLU AVG. ACC: {accuracy}')


if __name__ == "__main__":
    # print the average and standard deviation of the numbers
    # of tokens for each MMLU subject

    model = 'meta-llama/Meta-Llama-3.1-8B'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    subject_lens = {}
    for subject in MMLU_SUBJECTS:
        dataset = load_dataset('lukaemon/mmlu', subject, trust_remote_code=True)
        few_shots = get_fewshots(dataset, subject, shift=1)

        lens = []

        qanda = [
            format_mmlu_plain(question, subject, few_shots)
            for question in dataset["test"]
        ]

        for input_slice in qanda:
            text, truth = input_slice
            inputs = tokenizer(text, return_tensors='pt', truncation=False)

            inputs = inputs.input_ids
            lens.append(inputs.size(1))

        subject_lens[subject] = (np.mean(lens), np.std(lens))
        print(f"{subject=} mean: {np.mean(lens)} std: {np.std(lens)}")

    with open("./saves/llama_eval/mmlu-stats.json", "w") as f:
        json.dump(subject_lens, f)
