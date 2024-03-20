import gc
import os
from lightning.pytorch.strategies import DDPStrategy
import copy
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.onnx
import lightning as pl
import transformers
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from timber.models.umbc import SlotSetEncoder
import torch.utils.checkpoint
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed.checkpoint
import deepspeed
import torch.autograd
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import DeepSpeedStrategy

# torch.autograd.set_detect_anomaly(True)

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training

from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer

from timber.utils import seed
from timber.dataset.labdataset import LabDataset
from timber.dataset.booksum import BookSumDataset
from timber.dataset.alpaca import AlpacaDataset
from timber.dataset.openwebtext import OpenWebTextDataset
from torch.utils.data import DataLoader, random_split

torch.set_float32_matmul_precision('high')


@dataclass
class TrainConfig:
    dev_run: bool = False
    disable_kd: bool = False
    using_fsdp: bool = False
    lr: float = 5e-5
    batch_size: int = 1
    accumulation_steps: int = 2
    lora_r: int = 32
    save_steps: int = 100
    dense_queries: int = None
    seq_len: int = 1024
    max_steps: int = -1
    model_checkpoint_dir: str = "./checkpoint"
    dataset: str = 'wikitext2'
    load_from_checkpoint: str = None
    k: int = 512
    slots: int = 32
    block_size_q: int = 8
    block_size_k: int = 8
    init_from_checkpoint: str = None
    method: str = 'none'
    model: str = 'qwen500m'
    disable_global_context: bool = False
    window: int = 256


class LabDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config: TrainConfig,
        num_workers: int = 0,
        data_dir: Path = "data",
        download: bool = True,
        train_size: float = 0.9,
    ):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.block_size = config.seq_len
        self.download = download
        self.num_workers = num_workers
        self.train_size = train_size
        self.dataset = None
        self.tokenizer = load_tokenizer()
        self.bsize = config.batch_size

    def prepare_data(self):
        if self.config.dataset in ['wikitext2', 'wikitext103']:
            self.dataset = LabDataset(
                data_dir=self.data_dir,
                block_size=self.block_size,
                download=self.download,
                tokenizer=self.tokenizer,
                dataset=self.config.dataset,
            )
        elif self.config.dataset in ['alpaca']:
            self.dataset = AlpacaDataset(tokenizer=self.tokenizer, )
        elif self.config.dataset in ['booksum']:
            self.dataset = BookSumDataset(tokenizer=self.tokenizer, )
        elif self.config.dataset in ['openwebtext']:
            self.dataset = OpenWebTextDataset(
                tokenizer=self.tokenizer,
                stride=self.block_size,
            )
        else:
            raise Exception()

    def setup(self, stage: str):
        # safe to call this again because data will not be double downloaded
        # this needs to be called here in order to get called on every device
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        self.prepare_data()

        if self.config.dataset in ['wikitext2', 'wikitext103']:
            if stage == "fit" or stage is None:
                test_size = min(100, len(self.dataset) * (1 - self.train_size))
                train_size = int(len(self.dataset) - test_size)
                # to ensure they are both ints and sum to the total dataset size
                test_size = len(self.dataset) - train_size

                self.train_data, self.val_data = random_split(
                    self.dataset, lengths=[train_size, test_size])
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        elif self.config.dataset in ['booksum', 'alpaca', 'openwebtext']:
            if stage == "fit" or stage is None:

                def train_val_dataset(dataset, val_split=0.05):
                    train_idx, val_idx = train_test_split(list(
                        range(len(dataset))),
                                                          test_size=val_split)
                    train = Subset(dataset, train_idx)
                    valid = Subset(dataset, val_idx)
                    return train, valid

                self.train_data, self.val_data = train_val_dataset(
                    self.dataset)
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        else:
            raise Exception()

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          num_workers=self.num_workers,
                          batch_size=self.bsize)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          num_workers=self.num_workers,
                          batch_size=self.bsize)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          num_workers=self.num_workers,
                          batch_size=self.bsize)


def load_model(
    train_config: TrainConfig = None,
    method='timber',
    device='cpu',
):
    if train_config.using_fsdp:
        device = 'cpu'

    MODELS = {
        'llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'llama1.3b-share-gpt': 'princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT',
        'llama1.3b': 'princeton-nlp/Sheared-LLaMA-1.3B',
        'llama13b': 'meta-llama/Llama-2-13b-hf',
        'qwen7b': 'Qwen/Qwen1.5-7B-Chat',
        'qwen14b': 'Qwen/Qwen1.5-14B-Chat',
        'yi6b': '01-ai/Yi-6B-200K',
        'yi34b': '01-ai/Yi-34B-200K',
    }
    assert train_config.model in MODELS, MODELS.keys()
    model_id = MODELS[train_config.model]

    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    config._use_umbc = train_config.method == "umbc"
    config._umbc_slots = train_config.slots
    config._window = train_config.window

    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=[
            "sse_q",
            "sse_k",
            "sse_v",
            "slots",
            "norm_slots",
            "norm_after",
            # "input_layernorm",
            # "post_attention_layernorm",
            # "norm",
        ])
    if train_config.using_fsdp:
        quant_config = None

    # print(f"{quant_config=}")

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map={"": device} if device != 'cpu' else 'cpu',
        load_in_4bit=quant_config is None,
        quantization_config=quant_config,
        # torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    if train_config.method == "umbc":
        # bitsandbytes somehow did not ignore this module and I could never figure out why.
        # just reinit the parameters to get rid of large values and NaN's
        model.model.sse.slots.reset_parameters()
        model.model.sse.norm_slots.reset_parameters()
        model.model.sse.norm_after.reset_parameters()

    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = method
            m.tree_k = train_config.k
            m.tree_block_size_q = train_config.block_size_q
            m.tree_block_size_k = train_config.block_size_k
            if train_config.dense_queries is None:
                train_config.dense_queries = train_config.k
            m.tree_dense_queries = train_config.dense_queries
            m.tree_using_context_avg = not train_config.disable_global_context
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = True
            if train_config.using_fsdp:
                # m._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
            else:
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    if method not in ["none", "umbc"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=train_config.lora_r // 2,
            lora_dropout=0.05,
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
                'sse_q',
                "sse_k",
                "sse_v",
                "slots",
                "norm_slots",
                'norm_after',
            ])

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if train_config.init_from_checkpoint is not None:
            print('loading from', train_config.init_from_checkpoint)
            state_dict = torch.load(train_config.init_from_checkpoint,
                                    map_location='cpu')['state_dict']
            keys = list(state_dict.keys())
            for key in keys:
                x = state_dict[key]
                state_dict[key.strip('model.')] = x
                del state_dict[key]
            model.load_state_dict(state_dict)
            print('lora checkpoint loaded from',
                  train_config.init_from_checkpoint)

    return model


def load_tokenizer():
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return tokenizer


class LabModule(pl.LightningModule):

    def __init__(self, config: TrainConfig):
        super().__init__()

        self.validation_preds = []
        self.validation_targets = []

        self.config = config

    def forward(
        self,
        inputs,
        target,
        output_hidden_states=False,
        position_ids=None,
    ):
        return self.model(
            inputs,
            labels=target,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
        )

    def step_umbc(self, full_inputs, full_target, output_teacher):
        i = torch.randint(8, self.config.seq_len + 1, size=(1, )).item()
        cutoff = min(i, self.config.window - 1)
        start, end = i - cutoff, i + 1
        # print(f"\n\n{cutoff=} {start=} {end=}\n\n")

        inputs = full_inputs[:, :end]
        target = full_target[:, :end]

        # print(
        #     f"\n\n\niteration: {i} {inputs.size()=} {target.size()=}\n\n\n"
        # )

        output = self(inputs,
                      target,
                      output_hidden_states=not self.config.disable_kd)

        loss_kd_hidden, loss_kd_logits = 0, 0
        if not self.config.disable_kd:
            logits = output.logits
            target = target[:, start:end]

            for teacher_layer, student_layer in zip(
                    output_teacher.hidden_states, output.hidden_states):

                # print(
                #     f"sanity check student and teacher layer before slicing: {student_layer.size()=} {teacher_layer.size()=} {cutoff=}"
                # )
                student_layer = student_layer[:, -(cutoff + 1):]
                teacher_layer = teacher_layer[:, start:end]
                # print(
                #     f"sanity check student and teacher layer: {student_layer.size()=} {teacher_layer.size()=}"
                # )

                loss_kd_hidden += torch.nn.functional.mse_loss(
                    student_layer.to(torch.float32),
                    teacher_layer.to(torch.float32))
            loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)

            # output logits were already truncated within the model
            # print(
            #     f"sanity check out logits kd (before slicing): {output_teacher.logits.size()=} {output.logits.size()}"
            # )
            teacher_logits = output_teacher.logits[:, start:end]
            # print(
            #     f"sanity check out logits kd: {teacher_logits.size()=} {output.logits.size()}"
            # )

            # print(
            #     f"{output.logits.size()=} {logits.size()=} {teacher_logits.size()=}"
            # )

            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.reshape(-1, logits.shape[-1]).to(
                    torch.float32).log_softmax(-1),
                teacher_logits.reshape(-1, logits.shape[-1]).to(
                    torch.float32).softmax(-1),
                reduction='batchmean',
            )

        loss = output.loss
        if not self.config.disable_kd:
            loss = loss * 0.1 + (loss_kd_hidden + loss_kd_logits) * 2.5

        self.log_losses(loss, output.loss, loss_kd_hidden, loss_kd_logits)
        return loss

    def log_losses(self, total, ce, kd_hidden=None, kd_logits=None):
        self.log("train_loss_model", ce.item())
        if not self.config.disable_kd:
            if kd_hidden > 0:
                self.log("train_loss_kd_hidden", kd_hidden.item())
            if kd_logits > 0:
                self.log("train_loss_kd_logits", kd_logits.item())
        self.log("train_loss", total.item())

    def step_plain(self, inputs, target, output_teacher=None):
        output = self(inputs,
                      target,
                      output_hidden_states=not self.config.disable_kd)

        loss_kd_hidden, loss_kd_logits = 0, 0
        if not self.config.disable_kd:
            logits = output.logits
            for teacher_layer, student_layer in zip(
                    output_teacher.hidden_states, output.hidden_states):

                loss_kd_hidden += torch.nn.functional.mse_loss(
                    student_layer.to(torch.float32),
                    teacher_layer.to(torch.float32))
            loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)

            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.reshape(-1, logits.shape[-1]).to(
                    torch.float32).log_softmax(-1),
                output_teacher.logits.reshape(-1, logits.shape[-1]).to(
                    torch.float32).softmax(-1),
                reduction='batchmean',
            )

        loss = output.loss
        if not self.config.disable_kd:
            loss = loss * 0.1 + (loss_kd_hidden + loss_kd_logits) * 2.5

        self.log_losses(loss, output.loss, loss_kd_hidden, loss_kd_logits)
        return loss

    def training_step(self, batch, batch_idx):
        if self.teacher is not None:
            self.teacher.eval()
        self.model.train()

        inputs, target = batch
        output_teacher = None
        if not self.config.disable_kd:
            with torch.no_grad():
                output_teacher = self.teacher(
                    inputs, output_hidden_states=not self.config.disable_kd)

        if self.config.method == "umbc":
            loss = self.step_umbc(inputs, target, output_teacher)
            return loss

        loss = self.step_plain(inputs, target)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        inputs, target = batch

        if self.config.method == "umbc":
            with torch.no_grad():
                output_logits = torch.zeros(inputs.size(0),
                                            inputs.size(1),
                                            32000,
                                            device=inputs.device)

                for i in range(self.config.seq_len):
                    cutoff = min(i, self.config.window - 1)
                    start, end = i - cutoff, i + 1
                    # print(f"\n\n{cutoff=} {start=} {end=}\n\n")

                    _inputs = inputs[:, :end]
                    _target = target[:, start:end]

                    output = self(_inputs, _target)
                    output_logits[:, i] = output.logits[:, -1]

                logits = output_logits[:, :-1].reshape(-1,
                                                       output_logits.size(-1))
                t = target[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(logits, t)

        else:
            with torch.no_grad():
                output = self(inputs, target)
                loss, output_logits = output.loss, output.logits

        # evaluate both regular models and our model only on the last part of the windo
        self.log("val/loss", loss.item())

        self.validation_preds.append(output_logits[:, :-1].cpu())
        self.validation_targets.append(target[:, 1:].cpu())

    def on_validation_epoch_end(self):
        from torchmetrics.text.perplexity import Perplexity
        with torch.no_grad():

            device = f"cuda:{self.local_rank}"
            # device = 'cpu'
            # if self.config.using_fsdp:
            #     device = 'cuda'
            calculator = Perplexity().to(device)
            for preds, target in zip(self.validation_preds,
                                     self.validation_targets):
                calculator.update(preds.to(device), target.to(device))
            ppl = calculator.compute()
        ppl = ppl.item()
        self.log("val/ppl", ppl)

        self.validation_preds.clear()
        self.validation_targets.clear()

    def setup(self, stage):
        if stage == "fit":
            tconfig = copy.deepcopy(self.config)
            tconfig.method = "none"

            sconfig = copy.deepcopy(self.config)
            if self.config.method == "umbc":
                sconfig.seq_len = self.config.slots + self.config.window

            self.model = load_model(train_config=sconfig,
                                    method=sconfig.method,
                                    device=self.local_rank)
            if not self.config.disable_kd:
                self.teacher = load_model(train_config=tconfig,
                                          method='none',
                                          device=self.local_rank)
            else:
                self.teacher = None

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            # print(name, p.requires_grad, p.shape, p.dtype)
            if p.requires_grad:
                params.append(p)
        if self.config.using_fsdp:
            return DeepSpeedCPUAdam(params, lr=self.config.lr)
        # return DeepSpeedCPUAdam(params, lr=self.config.lr)
        return torch.optim.AdamW(params, lr=self.config.lr)


def main(config: TrainConfig):
    os.makedirs(config.model_checkpoint_dir, exist_ok=True)

    d = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if config.using_fsdp:
        devices = f"{d}"
        policy = {LlamaDecoderLayer}
        # strategy = FSDPStrategy(
        #     auto_wrap_policy=policy,
        #     activation_checkpointing_policy=policy,
        #     cpu_offload=True,
        # )
        # strategy = 'deepspeed_stage_3'
        deepspeed_config = {
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu"
                },
                "offload_optimizer": {
                    "device": "cpu"
                },
                "max_live_parameters": 5e8,
                "max_reuse_distance": 1e8,
                "contiguous_gradients": True,
                "overlap_comm": False,
                "allgather_bucket_size": 1e7,
                "reduce_bucket_size": 1e7,
            },
        }
        strategy = DeepSpeedStrategy(config=deepspeed_config)
    else:
        devices = f"{d}"
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
        # strategy = "ddp_find_unused_parameters_false"

    if config.method == 'timber':
        filename = f'llama32k-{config.dataset}-{config.seq_len}-bq{config.block_size_q}-bk{config.block_size_k}-k{config.k}-{{epoch:02d}}-{{step}}'
    elif config.method == 'none':
        filename = f'llama32k-{config.dataset}-{config.seq_len}-{{epoch:02d}}-{{step}}'
    elif config.method == 'umbc':
        filename = f'llama1.3b-{config.dataset}-{config.seq_len}-{{epoch:02d}}-{{step}}'
    elif config.method == 'reformer':
        filename = f'llama32k-{config.method}-{config.dataset}-{config.seq_len}-k{config.k}-{{epoch:02d}}-{{step}}'
    elif config.method == 'performer':
        filename = f'llama32k-{config.method}-{config.dataset}-{config.seq_len}-{{epoch:02d}}-{{step}}'
    else:
        raise Exception()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="step",
        mode="max",
        dirpath=config.model_checkpoint_dir,
        filename=filename,
        every_n_train_steps=config.save_steps,
        enable_version_counter=False,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '-'
    checkpoint_callback.FILE_EXTENSION = '.pth'

    trainer = pl.Trainer(
        log_every_n_steps=1,
        devices=devices,
        accelerator="gpu",
        strategy=strategy,
        # precision="bf16-mixed",
        precision="16-mixed",
        default_root_dir=config.model_checkpoint_dir,
        accumulate_grad_batches=config.accumulation_steps,
        max_epochs=20,
        check_val_every_n_epoch=20,
        max_steps=config.max_steps,
        logger=AimLogger(experiment="umbc-llm"),
        enable_checkpointing=True,
        fast_dev_run=config.dev_run,
        callbacks=[checkpoint_callback],
    )

    datamodule = LabDataModule(config=config)
    model = LabModule(config=config)

    # model = torch.compile(model, mode="reduce-overhead")
    kwargs = dict(
        model=model,
        datamodule=datamodule,
    )
    if config.load_from_checkpoint is not None:
        kwargs['ckpt_path'] = config.load_from_checkpoint
    trainer.fit(**kwargs)


if __name__ == "__main__":
    seed()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='llama32k', type=str)
    parser.add_argument('--using_fsdp', action='store_true')
    parser.add_argument('--disable_kd', action='store_true')
    parser.add_argument('--dev_run', action='store_true')
    parser.add_argument('--disable_global_context', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=-1, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--lora_r', default=-1, type=int)
    parser.add_argument('--lr', default=-1, type=float)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--seq_len', default=-1, type=int)
    parser.add_argument('--save_steps', default=-1, type=int)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--k', default=512, type=int)
    parser.add_argument('--slots', default=32, type=int)
    parser.add_argument('--block_size_q', default=16, type=int)
    parser.add_argument('--block_size_k', default=2, type=int)
    parser.add_argument('--method', default='umbc', type=str)
    parser.add_argument('--window', default=256, type=int)

    args = parser.parse_args()

    train_config = TrainConfig(
        dev_run=args.dev_run,
        using_fsdp=args.using_fsdp,
        disable_kd=args.disable_kd,
        dataset=args.dataset,
        load_from_checkpoint=args.checkpoint,
        k=args.k,
        block_size_q=args.block_size_q,
        block_size_k=args.block_size_k,
        method=args.method,
        model=args.model,
        disable_global_context=args.disable_global_context,
        window=args.window,
        slots=args.slots,
    )
    if args.gradient_accumulation_steps > 0:
        train_config.accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r > 0:
        train_config.lora_r = args.lora_r
    if args.lr > 0:
        train_config.lr = args.lr
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.max_steps > 0:
        train_config.max_steps = args.max_steps
    if args.seq_len > 0:
        train_config.seq_len = args.seq_len
    if args.save_steps > 0:
        train_config.save_steps = args.save_steps

    main(train_config)
