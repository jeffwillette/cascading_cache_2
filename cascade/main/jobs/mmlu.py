import json
import os
import time
import math
import torch
import transformers
from datasets import load_dataset
import tqdm
import numpy as np
import deepspeed

from cascade.utils import seed, get_bench
from cascade.main.jobs.pg19 import get_injection_policy

MMLU_FORMAT = """> The following are multiple choice questions (with answers) about {subject_name}.

{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

MMLU_FORMAT_TARGET_QUESTION = """> I want you to answer the following multiple choice question about {subject_name}.

{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:{answer_placeholder}"""

MMLU_REVERSED = """I want you to answer a question, but first I will show you some examples of similar questions. The question I am interested in is:

{original_question}

The examples of similar questions are:

{examples}

I want to know the answer to question 1. The answer ({answer_placeholder}) to the question I am interested in is:"""

MMLU_STEM = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'electrical_engineering',
    'elementary_mathematics',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_mathematics',
    'high_school_physics',
    'high_school_statistics',
    'machine_learning',
]

MMLU_SUBJECTS_SORTED = [
    'management',
    'global_facts',
    'abstract_algebra',
    'human_sexuality',
    'medical_genetics',
    'us_foreign_policy',
    'college_chemistry',
    'miscellaneous',
    'conceptual_physics',
    'world_religions',
    'jurisprudence',
    'human_aging',
    'anatomy',
    'business_ethics',
    'public_relations',
    'high_school_geography',
    'college_physics',
    'college_mathematics',
    'electrical_engineering',
    'machine_learning',
    'computer_security',
    'philosophy',
    'clinical_knowledge',
    'high_school_computer_science',
    'marketing',
    'logical_fallacies',
    'nutrition',
    'high_school_microeconomics',
    'college_biology',
    'high_school_macroeconomics',
    'sociology',
    'moral_disputes',
    'econometrics',
    'prehistory',
    'astronomy',
    'college_computer_science',
    'high_school_mathematics',
    'virology',
    'high_school_government_and_politics',
    'high_school_psychology',
    'international_law',
    'elementary_mathematics',
    'high_school_biology',
    'college_medicine',
    'formal_logic',
    'high_school_chemistry',
    'high_school_physics',
    'professional_psychology',
    'moral_scenarios',
    'professional_accounting',
    'high_school_statistics',
    'security_studies',
    'professional_medicine',
    'professional_law',
    'high_school_world_history',
    'high_school_us_history',
    'high_school_european_history',
]

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


def exam_mmlu(model, tokenizer: transformers.PreTrainedTokenizer, texts, args):

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

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        max_length=model.config.max_position_embeddings,
        truncation=True,
        padding=True,
    )

    # print(inputs.input_ids.shape)
    # inputs["input_ids"] = inputs["input_ids"][:, :50]
    # inputs["attention_mask"] = inputs["attention_mask"][:, :50]

    seq_lens = inputs.attention_mask[:text_len].sum(dim=-1).tolist()

    with torch.no_grad():
        sink = True
        if sink:
            input_ids = inputs["input_ids"].cuda()

            # these are the first outputs after the last of the prompt sequence
            pred_indices = inputs.attention_mask[:text_len].sum(
                dim=-1, keepdim=True) - 1

            output = model(input_ids, use_cache=False)
            logits = output.logits[:text_len]

            pred_indices = pred_indices.unsqueeze(-1).repeat(
                1, 1, logits.size(-1))

            logits = torch.gather(logits.cpu(), 1, pred_indices)
            output = logits.softmax(dim=-1)

        else:
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


def format_mmlu_reversed(question, subject_name, few_shots):
    original_question = format_mmlu(
        question,
        1,
        subject_name,
        target_question=True,
        answer_placeholder="<|YOUR ANSWER|>",
    )

    examples = "\n\n".join(few_shots)
    truth = question['target']

    text = MMLU_REVERSED.format(
        original_question=original_question,
        examples=examples,
        answer_placeholder="<|YOUR ANSWER|>",
    )
    return text, truth


def format_mmlu_plain(question, subject_name, few_shots):
    text = format_mmlu(
        question,
        len(few_shots) + 1,
        subject_name,
        target_question=True,
        answer_placeholder="",
    )
    truth = question['target']
    text = "\n\n".join(few_shots + [
        text,
    ])
    return text, truth


def evaluate_mmlu(args, model, tokenizer, subject_name, reverse=False):
    dataset = load_dataset('lukaemon/mmlu', subject_name)

    few_shots = get_fewshots(dataset, subject_name, shift=2 if reverse else 1)

    t_start = time.time()
    results = []
    n_correct = 0
    seq_len_sum = 0

    qanda = [
        format_mmlu_reversed(question, subject_name, few_shots)
        if reverse else format_mmlu_plain(question, subject_name, few_shots)
        for question in dataset["test"]
    ]

    print("total len of questions and answers: ", len(qanda))
    n = math.ceil(len(qanda) / args.batch_size)

    with tqdm.tqdm(range(n), dynamic_ncols=True, leave=True,
                   desc=subject_name) as pbar:
        for i in pbar:
            input_slice = qanda[i * args.batch_size:(i + 1) * args.batch_size]

            texts = [v[0] for v in input_slice]
            truths = [v[1] for v in input_slice]

            batch_estimations, seq_lens = exam_mmlu(model, tokenizer, texts,
                                                    args)
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
    print(
        f'{subject_name} = Accuracy: {accuracy:.4f} %, avg_seq_len: {avg_seq_len:.2f}. elapsed: {elapsed:.1f} s'
    )

    os.makedirs('./saves/llama_eval/mmlu/', exist_ok=True)
    json_path = f'./saves/llama_eval/mmlu/{subject_name}_{args.model}_{args.method}_reverse_{reverse}.json'
    if args.method == 'sink':
        json_path = f'./saves/llama_eval/mmlu/{subject_name}_{args.model}_{args.method}_window_{args.window}_' + \
            f'head_reduction_{args.head_reduction}_cascade_{args.cascades}_sinks_{args.sinks}_' + \
            f'comment_{args.comment}_reverse_{reverse}.json'

    with open(json_path, 'w') as f:
        json.dump(
            {
                'accuracy': accuracy,
                'avg_seq_len': avg_seq_len,
                'elapsed': elapsed,
                # 'k': args.k,
                'model': args.model,
                # 'block_size_q': args.block_size_q,
                # 'block_size_k': args.block_size_k,
                # 'dense_queries': args.dense_queries,
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

    if args.method == "vanilla":
        model = model.cuda()

    elif args.method == "sink":
        model.model.setup_caches(args.world_size)

        if args.world_size > 1:
            model = deepspeed.init_inference(
                model,
                tensor_parallel={"tp_size": args.world_size},
                replace_with_kernel_inject=False,
                dtype=args.infer_dtype,
                injection_policy=get_injection_policy(args.model),
            )
        else:
            model = model.cuda()

    accuracies, reversed_accuracies = [], []
    # for subjects in MMLU_SUBJECTS:
    for subjects in MMLU_STEM:
        acc = evaluate_mmlu(args, model, tokenizer, subjects, reverse=False)
        accuracies.append(acc)

        # rev_acc = evaluate_mmlu(args, model, tokenizer, subjects, reverse=True)
        # reversed_accuracies.append(rev_acc)

    accuracy = np.array(accuracies).mean()
    print(f'MMLU AVG. ACC: {accuracy}')

    # rev_accuracy = np.array(reversed_accuracies).mean()
    # print(f'MMLU AVG. ACC: {rev_accuracy}')
