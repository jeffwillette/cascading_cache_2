# coding=utf-8
import types
from torch import nn
import math
from typing import Optional, Tuple, Union

from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np

from transformers.cache_utils import Cache
from cascade.models.flash_attention import attention
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.utils import logging
from cascade.models.cascading_cache import CascadingKVCache

from transformers.generation.logits_process import LogitsProcessorList

from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_one(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class CascadeAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:

        cascade = isinstance(past_key_value, CascadingKVCache)
        if cascade:
            return self.forward_cascade(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                use_flash=hidden_states.size(1) >= 64,
                position_embeddings=position_embeddings,
            )

        elif hasattr(self, "hyper_attn"):
            return self.forward_hyper_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        raise ValueError("unknown custom attention")

    # @torch.compile
    def qkv(self, x, size, kv_size):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(*size).transpose(1, 2)
        k = k.view(*kv_size).transpose(1, 2)
        v = v.view(*kv_size).transpose(1, 2)
        return q, k, v

    def forward_cascade(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        use_flash: bool = True,
        homogeneous_heads: bool = False,
        do_og_pos: bool = False,
        use_selfextend: bool = False,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        not homogeneous + og_pos = independent head
        homogeneous + not og_pos = original
        homogeneous + og_pos = streaming llm with original pos
        not homogeneous + not og_pos = naive
        """
        bsz, q_len, _ = hidden_states.size()

        # this will be set in the eager/flash methods
        if not hasattr(self, "last_q_size"):
            self.last_q_size = None

        # project qkv first for this layer
        query_states, key_states, value_states = self.qkv(
            hidden_states, (bsz, q_len, self.num_heads, self.head_dim),
            (bsz, q_len, self.num_key_value_heads, self.head_dim))

        first_it = past_key_value.seen_tokens_by_layer[self.layer_idx] == 0

        sink_key_states, sink_value_states, sink_pos, sink_mask, k_states, v_states, key_pos, key_mask, og_pos =\
            past_key_value.get_vals(self.layer_idx)

        max_pos = 0  # may be reset later
        if first_it:
            self.last_attn_called = None
            self.query_pos = torch.arange(0,
                                          query_states.size(2),
                                          device=query_states.device,
                                          dtype=torch.long).view(1, -1)

        if not first_it:
            if use_selfextend:
                og_pos = og_pos[:, 0, :]
                n_pos = og_pos[:, :past_key_value.window_length]
                g_pos = og_pos[:, past_key_value.window_length:]

                g_pos = g_pos // 3 + past_key_value.num_sink_tokens
                n_pos = n_pos - n_pos.amin(dim=-1, keepdim=True) + g_pos.amax(
                    dim=-1, keepdim=True) + 1
                og_pos = torch.cat((n_pos, g_pos), dim=-1)
                max_pos = torch.cat(
                    (sink_pos.amax().view(1), og_pos.amax().view(1)),
                    dim=-1).amax() + 1
                key_pos = og_pos
            else:
                if do_og_pos:
                    max_pos = torch.cat(
                        (sink_pos.amax().view(1), og_pos.amax().view(1)),
                        dim=-1).amax() + 1
                else:
                    max_pos = torch.cat(
                        (sink_pos.amax().view(1), key_pos.amax().view(1)),
                        dim=-1).amax() + 1

        query_pos = self.query_pos[:, :query_states.size(-2)] + max_pos

        query_states = self.rope(query_states, query_pos)
        key_states_pos = self.rope(key_states, query_pos)
        sink_key_states = self.rope(sink_key_states, sink_pos)

        if do_og_pos:
            b, h, s, d = k_states.size()
            k_states = self.rope(k_states.view(b * h, 1, s, d), og_pos.view(b * h, -1))
            k_states = k_states.view(b, h, s, d)
        else:
            k_states = self.rope(k_states, key_pos)

        key_states_pos = repeat_kv(key_states_pos, self.num_key_value_groups)
        sink_key_states = repeat_kv(sink_key_states, self.num_key_value_groups)
        sink_value_states = repeat_kv(sink_value_states, self.num_key_value_groups)
        k_states = repeat_kv(k_states, self.num_key_value_groups)
        v_states = repeat_kv(v_states, self.num_key_value_groups)
        scale = 1 / math.sqrt(query_states.size(-1))

        func_args = (
            query_states, key_states, value_states,
            key_states_pos, v_states, sink_key_states,
            sink_value_states, k_states, sink_mask,
            key_mask, first_it, past_key_value,
            scale, q_len, homogeneous_heads
        )

        if not use_flash:
            out, past_key_value = self.cascade_attn_eager(*func_args)
        else:
            out, past_key_value = self.cascade_attn_flash(*func_args)

        out = out.transpose(1, 2).contiguous()
        out = out.view(bsz, q_len, -1)
        out = self.o_proj(out)

        return out, None, past_key_value

    def cascade_attn_flash(
        self, query_states, key_states, value_states,
        key_states_pos, v_states, sink_key_states,
        sink_value_states, k_states, sink_mask,
        key_mask, first_it, past_key_value,
        scale, q_len, homogeneous_heads
    ):
        if first_it or self.last_attn_called != "flash" or self.last_q_size != query_states.size(2):
            self.last_attn_called = "flash"
            self.last_q_size = query_states.size(2)
            causal_mask = torch.full((query_states.size(2), query_states.size(2)), 1,
                                     device=query_states.device,
                                     dtype=torch.bool).triu(1)

            sink_mask = torch.full((query_states.size(2), sink_key_states.size(2)), 1,
                                   device=query_states.device,
                                   dtype=torch.bool)

            cache_mask = torch.full((query_states.size(2), k_states.size(2)), 1,
                                    device=query_states.device,
                                    dtype=torch.bool)

            self.causal_mask = torch.cat((sink_mask, cache_mask, causal_mask), dim=-1)

        n_sink = sink_key_states.size(2)
        self.causal_mask[:, :n_sink] =\
            self.causal_mask[:, :n_sink] * sink_mask[0, 0]

        self.causal_mask[:, n_sink:n_sink + k_states.size(2)] = \
            self.causal_mask[:, n_sink:n_sink + k_states.size(2)] * key_mask[0, 0]

        out, scores = attention(
            query_states,
            torch.cat((sink_key_states, k_states, key_states_pos), dim=2),
            torch.cat((sink_value_states, v_states, repeat_kv(value_states, self.num_key_value_groups)), dim=2),
            True, scale, self.causal_mask, past_key_value.beta,
        )

        ema_scores, scores = self.calc_scores(scores[:, :, n_sink:], homogeneous_heads, k_states.size(2), q_len)

        beta = past_key_value.beta
        past_key_value.score_cache[self.layer_idx] = beta**out.size(
            2) * past_key_value.score_cache[self.layer_idx] + ema_scores

        _ = past_key_value.update(
            key_states, value_states, self.layer_idx, score_states=scores)

        return out, past_key_value

    def calc_scores(self, scores, homogeneous_heads, n_keys, q_len):
        kvh = self.num_key_value_heads
        if homogeneous_heads:
            if self.config._head_reduction == "mean":
                ema_scores = scores[:, :, :n_keys].mean(dim=1, keepdim=True)
                scores = scores[:, :, -q_len:].mean(dim=1, keepdim=True).repeat(1, kvh, 1)
            elif self.config._head_reduction == "max":
                ema_scores = scores[:, :, :n_keys].amax(dim=1, keepdim=True)
                scores = scores[:, :, -q_len:].amax(dim=1, keepdim=True).repeat(1, kvh, 1)
            elif self.config._head_reduction == "median":
                ema_scores = scores[:, :, :n_keys].median(dim=1, keepdim=True).values
                scores = scores[:, :, -q_len:].median(dim=1, keepdim=True).values.repeat(1, kvh, 1)
        else:
            b, h, s = scores.size()
            ema_scores = scores[:, :, :n_keys].view(b, kvh, h // kvh, -1).mean(dim=2)
            scores = scores[:, :, -q_len:].view(b, kvh, h // kvh, -1).mean(dim=2)

        return ema_scores, scores

    def cascade_attn_eager(
        self,
        query_states, key_states, value_states,
        key_states_pos, v_states, sink_key_states,
        sink_value_states, k_states, sink_mask,
        key_mask, first_it, past_key_value,
        scale, q_len, homogeneous_heads,
    ):
        val = torch.finfo(query_states.dtype).min
        if first_it or self.last_attn_called != "eager" or self.last_q_size != query_states.size(2):
            self.last_attn_called = "eager"
            self.last_q_size = query_states.size(2)

            self.causal_mask = torch.full(
                (query_states.size(2), query_states.size(2)),
                True,
                device=query_states.device,
                dtype=torch.bool).triu(1)

            # set the ema mask
            out = []
            for i in range(q_len):
                beta = past_key_value.beta
                exps = (1 - beta) * beta**torch.arange(
                    q_len - i,
                    device=query_states.device,
                    dtype=query_states.dtype)
                out.append(
                    torch.cat((torch.zeros(i,
                                           device=query_states.device,
                                           dtype=query_states.dtype), exps)))

            self.ema_mask = torch.stack(out).T

        qattn = torch.einsum("bhqd,bhkd->bhqk", query_states * np.sqrt(scale), key_states_pos * np.sqrt(scale))
        qattn = qattn + (self.causal_mask * val).half()

        sattn = torch.einsum("bhqd,bhkd->bhqk", query_states * np.sqrt(scale), sink_key_states * np.sqrt(scale))
        sattn = sattn + (val * sink_mask[:1, :1, None, :]).half()

        cattn = torch.einsum("bhqd,bhkd->bhqk", query_states * np.sqrt(scale), k_states * np.sqrt(scale))
        cattn = cattn + (val * key_mask[:1, :1, None, :]).half()

        attn = torch.cat((sattn, cattn, qattn), dim=-1)
        attn = attn.softmax(dim=-1)

        sk, k = sink_key_states.size(2), k_states.size(2)
        out = attn[:, :, :, :sk] @ sink_value_states
        out += attn[:, :, :, sk:sk + k] @ v_states
        out += attn[:, :, :, -q_len:] @ repeat_kv(value_states, self.num_key_value_groups)

        beta = past_key_value.beta
        exps = (1 - beta) * beta**torch.arange(
            attn.size(2), device=attn.device).flip(dims=(0, ))

        ema_scores = (attn[:, :, :, sk:sk + k] * \
                      exps[None, None, :, None]).sum(dim=2).half()

        scores = attn[:, :, :, -q_len:] * self.ema_mask[None, None, :].half()
        scores = scores.sum(dim=2)
        scores = torch.cat((ema_scores, scores), dim=-1)

        ema_scores, scores = self.calc_scores(
            scores, homogeneous_heads, k_states.size(2), q_len)

        past_key_value.score_cache[self.layer_idx] = beta**scores.size(2) * \
            past_key_value.score_cache[self.layer_idx] + ema_scores

        scores = scores.repeat(1, query_states.size(1), 1)

        _ = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            score_states=scores,
        )
        return out, past_key_value

    def forward_hyper_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            key_states, value_states = past_key_value.update(
                key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.hyper_attn(query_states,
                                      key_states,
                                      value_states,
                                      causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class LlamaCascadeAttention(CascadeAttention, LlamaAttention):
    def rope(self, x, pos):
        cos, sin = self.rotary_emb(x, pos)
        out = apply_rotary_pos_emb_one(x, cos, sin)
        return out


class Qwen2CascadeAttention(CascadeAttention, Qwen2Attention):
    def rope(self, x, pos):
        # necessary because Qwen expects a max index and our index could be larger than the size
        # since we do not have the sink token positions in the cascade part of the kv cache
        cos, sin = self.rotary_emb(x, 32768)

        cos = cos[pos.view(-1)].reshape(1, pos.size(1), cos.size(-1))
        sin = sin[pos.view(-1)].reshape(1, pos.size(1), sin.size(-1))

        out = apply_rotary_pos_emb_one(x, cos, sin)
        return out


def _sample_monkeypatch(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    logits_warper: Optional[LogitsProcessorList],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
            `generation_config`)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    first_it = True
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # because cascade cache is on every layer
        # model_inputs["past_key_values"] = None
        model_inputs["attention_mask"] = None
        model_inputs["position_ids"] = None
        model_inputs["cache_position"] = None
        # model_inputs["use_cache"] = True

        if not first_it:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1:]

        # print(f"{model_inputs=}")
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        # forward pass to get next token

        if first_it:

            # mdl = self.model
            # stride = mdl.config._cascade_stride
            # window = mdl.config._window // mdl.config._cascades
            # max_seq_len = mdl.config._window

            # past_key_values = CascadingKVCache(
            #     window,
            #     num_sink_tokens=mdl.config._sinks,
            #     max_batch_size=mdl.config._batch_size,
            #     heads=mdl.config.num_key_value_heads // mdl.config.world_size,
            #     dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
            #     max_seq_len=max_seq_len,
            #     dtype=torch.float16,
            #     device=mdl.embed_tokens.weight.device,
            #     cascade_func=mdl.config._cascade_func,
            #     head_reduction=mdl.config._head_reduction,
            #     layers=len(mdl.layers),
            # )
            # model_inputs["past_key_values"] = past_key_values
            inputs = model_inputs["input_ids"]

            for i in range(0, inputs.size(1), stride):
                print(i)
                model_inputs["input_ids"] = inputs[:, i:i + stride]
                outputs = self(**model_inputs, return_dict=True)

            first_it = False
        else:
            outputs = self(**model_inputs, return_dict=True)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone()

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def sample_monkeypatch(model) -> nn.Module:
    model._sample = types.MethodType(_sample_monkeypatch, model)
    return model


if __name__ == "__main__":
    from cascade.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
    from cascade.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
    from argparse import Namespace
    import transformers

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

    for model in ["llama3.1-8b", "qwen7b"]:
        args = Namespace(
            model=model,
            batch_size=1,
            sinks=4,
            cascades=4,
            window=2048,
            world_size=1,
            cascade_func="pow2",
            head_reduction="mean",
            method="sink",
            cascade_stride=128,
        )

        device = 'cuda:0'
        MODELS = {
            'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B',
            'qwen7b': 'Qwen/Qwen1.5-7B',
        }
        model_id = MODELS[args.model]

        config = get_config(model_id)

        config._attn_implementation = config.attn_implementation = 'eager'
        config._batch_size = args.batch_size
        config._sinks = args.sinks
        config._cascades = args.cascades
        config._window = args.window
        config.world_size = args.world_size
        config._cascade_func = args.cascade_func
        config._head_reduction = args.head_reduction
        config._method = args.method
        config._cascade_stride = args.cascade_stride

        print(f"{config=}")

        args.infer_dtype = torch.float16
        from_pretrained_kwargs = dict(
            config=config,
            device_map={"": device},
            # quantization_config=transformers.BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=args.infer_dtype,
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

        model = get_model(model_id, **from_pretrained_kwargs)
        model = sample_monkeypatch(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        inputs = torch.randint(0, 1024, size=(1, 2048), device=device)

        mdl = model.model
        stride = mdl.config._cascade_stride
        window = mdl.config._window // mdl.config._cascades
        max_seq_len = mdl.config._window

        past_key_values = CascadingKVCache(
            window,
            num_sink_tokens=mdl.config._sinks,
            max_batch_size=mdl.config._batch_size,
            heads=mdl.config.num_key_value_heads // mdl.config.world_size,
            dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
            max_seq_len=max_seq_len,
            dtype=torch.float16,
            device=mdl.embed_tokens.weight.device,
            cascade_func=mdl.config._cascade_func,
            head_reduction=mdl.config._head_reduction,
            layers=len(mdl.layers),
        )

        print("testing cascade attention iterative loop")
        for i in range(0, inputs.size(1), args.cascade_stride):
            inp = inputs[:, i: i + args.cascade_stride]
            output = model(inp, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            del output

        print("testing generate method")
        past_key_values.reset(verbose=True)

        text = "Are you conscious? " * 300
        inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        output = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            past_key_values=past_key_values,
        )[0]

        pred = tokenizer.decode(output[inputs.size(1):])

        print("finished")
