# coding=utf-8
from torch import nn
import math
from typing import Optional, Tuple

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
        if first_it or self.last_attn_called != "flash":
            self.last_attn_called = "flash"
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
        if first_it or self.last_attn_called != "eager":
            self.last_attn_called = "eager"
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
