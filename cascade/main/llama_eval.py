import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import TextStreamer

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from cascade.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from cascade.models.qwen.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from cascade.utils import seed, get_bench

# from cascade.main.jobs.bench_single_layer import job_bench_single_layer
from cascade.main.jobs.ppl import job_ppl
from cascade.main.jobs.passkey import job_passkey
from cascade.main.jobs.latency import job_latency
from cascade.main.jobs.ppl_memory import job_ppl_memory
from cascade.main.jobs.pg19 import job_ppl_pg19
from cascade.main.jobs.profile import job_profile
from cascade.main.jobs.pg19_compile import job_ppl_pg19_compile
# from cascade.main.jobs.stream import job_stream
from cascade.main.jobs.mmlu import job_mmlu
from cascade.main.eval_args import eval_args, ArgsType


def get_model(model_id, **from_pretrained_kwargs):
    if "llama" in model_id.lower():
        return LlamaForCausalLM.from_pretrained(model_id,
                                                **from_pretrained_kwargs)
    elif "qwen" in model_id.lower():
        return Qwen2ForCausalLM.from_pretrained(model_id,
                                                **from_pretrained_kwargs)

    else:
        raise ValueError(f"{model_id} not supported")


def get_config(model_id):
    if "llama" in model_id.lower():
        return LlamaConfig.from_pretrained(model_id)
    elif "qwen" in model_id.lower():
        return Qwen2Config.from_pretrained(model_id)
    else:
        raise ValueError(f"{model_id} not supported")


def load_vllm_model(args: ArgsType):
    from vllm import LLM

    device = 'cuda:0'
    MODELS = {
        'vllm_llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'vllm_llama128k': 'NousResearch/Yarn-Llama-2-7b-128k',
        'vllm_llama13b_128k': 'NousResearch/Yarn-Llama-2-13b-128k',
        'vllm_llama100k': 'Yukang/Llama-2-7b-longlora-100k-ft',
        'vllm_llama32k_instruct': 'togethercomputer/Llama-2-7B-32K-Instruct',
        'vllm_llama1b': 'princeton-nlp/Sheared-LLaMA-1.3B',
        'vllm_llama7b': 'meta-llama/Llama-2-7b-hf',
        'vllm_llama13b': 'meta-llama/Llama-2-13b-hf',
        'vllm_qwen7b': 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
        'vllm_qwen14b': 'Qwen/Qwen1.5-14B-Chat',
        'vllm_qwen0.5b': 'Qwen/Qwen1.5-0.5B-Chat',
        'vllm_pythia70m': 'EleutherAI/pythia-70m',
        'vllm_yi6b': '01-ai/Yi-6B-200K',
        'vllm_yi34b': 'brucethemoose/Yi-34B-200K-RPMerge',
        'vllm_mixtral8x7b': 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
    }
    assert args.model in MODELS
    assert args.job in ['stream']
    model_id = MODELS[args.model]

    assert args.checkpoint is None

    seq_len = args.stride
    # seq_len = 10600
    model = LLM(
        model_id,
        max_context_len_to_capture=seq_len,
        max_num_seqs=args.batch_size,
        max_model_len=seq_len,
        swap_space=0,
        kv_cache_dtype='fp8_e5m2',
        dtype='half',
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=os.environ.get('FORCE_EAGER', '0') == '1',
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer, device


def get_dtype(model_name):
    if "llama" in model_name.lower():
        return torch.float16
    elif "qwen" in model_name.lower():
        return torch.float16
    else:
        raise ValueError(f"unknown dtype for model: {model_name}")


def load_model(args):
    if args.model.startswith('vllm'):
        return load_vllm_model(args)

    device = 'cuda:0'
    MODELS = {
        'llama3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B',
        'llama7b': 'togethercomputer/LLaMA-2-7B-32K',
        'llama13b': 'meta-llama/Llama-2-13b-hf',
        'llama13b_32k': 'Yukang/Llama-2-13b-longlora-32k-ft',
        'llama7b-chat': '/d1/dataset/llama/models/llama_v2/llama-2-7b-chat-hf',
        "llama2-7b-chat-32k": "togethercomputer/Llama-2-7B-32K-Instruct",
        'qwen14b': 'Qwen/Qwen1.5-14B',
        'qwen7b': 'Qwen/Qwen1.5-7B',
        'qwen7b-chat': 'Qwen/Qwen1.5-7B-Chat',
        'qwen0.5b': 'Qwen/Qwen1.5-0.5B',
        'llama1.3b': 'princeton-nlp/Sheared-LLaMA-1.3B',
        'llama3-8b-instruct':
        "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-8B-Instruct",
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3-70b-instruct':
        "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-70B-Instruct",
        'llama2-70b': "/d1/dataset/llama/models/llama_v2/llama-2-70b",
    }

    assert args.model in MODELS, MODELS.keys()
    model_id = MODELS[args.model]

    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))

    config = get_config(model_id)

    config._attn_implementation = config.attn_implementation = 'eager'
    if args.method == "vanilla":
        config._attn_implementation = config.attn_implementation = 'flash_attention_2'

    if args.job == "latency":
        config.max_position_embeddings = 2 ** 19

    config._batch_size = args.batch_size
    config._sinks = args.sinks
    config._cascades = args.cascades
    config._window = args.window
    config.world_size = args.world_size
    config._cascade_func = args.cascade_func
    config._head_reduction = args.head_reduction
    config._method = args.method

    if args.model == "llama13b_32k":
        config.max_position_embeddings = 32768

    print(f"{config=}")

    if "70B" not in model_id or "70b" not in model_id:
        args.infer_dtype = get_dtype(model_id)
        from_pretrained_kwargs = dict(
            config=config,
            # device_map={"": device},
            device_map=None,
            # quantization_config=transformers.BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=infer_dtype,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     llm_int8_skip_modules=[
            #         # "input_layernorm",
            #         # "post_attention_layernorm",
            #         # "norm",
            #     ]),
            torch_dtype=args.infer_dtype,
            trust_remote_code=True,
        )
    else:
        args.infer_dtype = get_dtype(model_id)
        from_pretrained_kwargs = dict(
            config=config,
            # device_map={"": device},
            device_map="auto",
            # load_in_4bit=True,
            # device_map=None,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=args.infer_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=args.infer_dtype,
            trust_remote_code=True,
        )

    if args.method == "vanilla":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **from_pretrained_kwargs)
    elif args.method == "sink":
        model = get_model(model_id, **from_pretrained_kwargs)
    elif args.method == "hyper":
        model = get_model(model_id, **from_pretrained_kwargs)
    else:
        raise ValueError("unsupported method")

    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = args.method

    if args.lora_r > 0 and args.checkpoint is not None:
        print("LoRA init")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args.lora_r,
            lora_alpha=args.lora_r // 2,
            lora_dropout=0.0,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                'input_layernorm',
                'post_attention_layernorm',
            ])

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        state_dict = torch.load(args.checkpoint,
                                map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            x = state_dict[key]
            state_dict[key.strip('model.')] = x
            del state_dict[key]

        result = model.load_state_dict(state_dict, strict=True)
        print('load result', result)
        print('lora checkpoint loaded from', args.checkpoint)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, device


def main():
    seed(seed=42)

    args = eval_args()

    assert args.job in [
        'ppl', 'ppl-pg19', 'ppl-memory', 'stream', 'mmlu',
        'bench_single_layer', 'passkey', 'profile', "latency"
    ]

    model, tokenizer, device = load_model(args)

    if args.job == 'ppl':
        raise NotImplementedError(
            "implementation needs to be updated to current")
        # job_ppl(args, model, tokenizer, device)
    elif args.job == 'ppl-memory':
        job_ppl_memory(args, model, tokenizer, device)
    elif args.job == 'latency':
        job_latency(args, model, tokenizer, device)
    elif args.job == 'passkey':
        job_passkey(args, model, tokenizer, device)
    elif args.job == 'ppl-pg19':
        # job_ppl_pg19(args, model, tokenizer, device)
        job_ppl_pg19_compile(args, model, tokenizer, device)
    elif args.job == 'profile':
        job_profile(args, model, tokenizer, device)
    elif args.job == 'stream':
        raise NotImplementedError(
            "implementation needs to be updated to current")
        # job_stream(args, model, tokenizer, device)
    elif args.job == 'mmlu':
        job_mmlu(args, model, tokenizer, device)
    elif args.job == 'bench_single_layer':
        raise NotImplementedError(
            "implementation needs to be updated to current")
        # job_bench_single_layer(args, model, tokenizer, device)
    else:
        raise Exception()


if __name__ == '__main__':
    main()
