import transformers
import math
import numpy as np
from typing import Optional, Tuple

import torch
from cascade.models.flash_attention import attention
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F
from cascade.models.cascade_attention import sample_monkeypatch

# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    rotate_half,
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2ForCausalLM
import types

__all__ = ["H2OLlamaForCausalLM", "H2OLlamaAttention",
           'H2OLlamaAttention_streaming', 'H2OLlamaForCausalLM_streaming']


from transformers.configuration_utils import PretrainedConfig

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
        bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_qwen(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, num_new_tokens=None):

        self._update_hh_score(attn_score_cache, num_new_tokens=num_new_tokens)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size + num_coming]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size + num_coming, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score = self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache, num_new_tokens=None):
        if num_new_tokens is None:
            num_new_tokens = attn_score_cache.shape[2]

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None


class H2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
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

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # original modded to fit llama 3 style
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]

        # Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        past_key_value = self.kv_cache(past_key_value[self.layer_idx], attn_weights.detach().clone())

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention(config)


def attn_batched_quad(query_states, key_states, value_states):
    N, HEAD, TDST, HID = query_states.shape
    N, HEAD_KV, TSRC, HID = key_states.shape
    assert key_states.shape == value_states.shape

    attn_weight_accumulator = torch.zeros(
        (N, HEAD, 1, TSRC),
        dtype=torch.float32,
        device=query_states.device
    )

    chunk_count = math.ceil((TDST * TSRC) / (2040 * 2048))
    chunk_size = math.ceil(TDST / chunk_count)

    attn_output_final = torch.empty_like(query_states)

    for i_tdst_start in range(0, TDST, chunk_size):
        i_tdst_end = min(i_tdst_start + chunk_size, TDST)
        attn_weights = torch.matmul(
            query_states[:, :, i_tdst_start:i_tdst_end] / np.sqrt(np.sqrt(HID)),
            key_states[:, :, :i_tdst_end + TSRC - TDST].transpose(2, 3) / np.sqrt(np.sqrt(HID))
        )
        if TDST > 1:
            idx_tdst = torch.arange(i_tdst_start, i_tdst_end, device=query_states.device)
            idx_tsrc = torch.arange(0, i_tdst_end, device=query_states.device) + TSRC - TDST
            attn_weights = torch.where(
                idx_tsrc[None, None, None, :] <= idx_tdst[None, None, :, None],
                attn_weights,
                -32000.0
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weight_accumulator[:, :, :, :i_tdst_end + TSRC - TDST].add_(attn_weights.sum(2, keepdim=True))
        attn_output = torch.matmul(attn_weights, value_states[:, :, :i_tdst_end + TSRC - TDST])

        attn_output_final.index_copy_(
            dim=2,
            index=torch.arange(i_tdst_start, i_tdst_end, device=attn_output.device),
            source=attn_output.to(attn_output_final.dtype)
        )

    attn_weights = attn_weight_accumulator
    attn_output = attn_output_final
    return attn_weights, attn_output


def attn_batched_full(query_states, key_states, value_states):
    N, HEAD, TDST, HID = query_states.shape
    N, HEAD_KV, TSRC, HID = key_states.shape
    assert key_states.shape == value_states.shape

    attn_weight_accumulator = torch.zeros(
        (N, HEAD, 1, TSRC),
        dtype=torch.float32,
        device=query_states.device
    )

    chunk_count = math.ceil((TDST * TSRC) / (2040 * 2048))
    chunk_size = math.ceil(TDST / chunk_count)

    attn_output_final = torch.empty_like(query_states)

    num_new_tokens = 0
    for i_tdst_start in range(0, TDST, chunk_size):
        i_tdst_end = min(i_tdst_start + chunk_size, TDST)

        attn_weights = torch.matmul(
            query_states[:, :, i_tdst_start:i_tdst_end] / np.sqrt(np.sqrt(HID)),
            key_states.transpose(2, 3) / np.sqrt(np.sqrt(HID))
        )
        if TDST > 1:
            idx_tdst = torch.arange(i_tdst_start, i_tdst_end, device=query_states.device) + TSRC - TDST
            idx_tsrc = torch.arange(0, TSRC, device=query_states.device)

            attn_weights = torch.where(
                idx_tsrc[None, None, None, :] <= idx_tdst[None, None, :, None],
                attn_weights,
                -32000.0
            )
        attn_weights = nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32
        ).to(query_states.dtype)

        num_new_tokens += attn_weights.size(2)
        # print(f"{attn_weight_accumulator.size()=} {attn_weights.size()=}")
        attn_weight_accumulator.add_(attn_weights.sum(2, keepdim=True))
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output_final.index_copy_(
            dim=2,
            index=torch.arange(i_tdst_start, i_tdst_end, device=attn_output.device),
            source=attn_output.to(attn_output_final.dtype)
        )

    attn_weights = attn_weight_accumulator
    attn_output = attn_output_final
    return attn_weights, attn_output, num_new_tokens


def attn_batched(query_states, key_states, value_states):
    N, HEAD, TDST, HID = query_states.shape
    N, HEAD_KV, TSRC, HID = key_states.shape
    assert key_states.shape == value_states.shape

    num_new_tokens = TDST

    mask = torch.zeros(TSRC - TDST, dtype=torch.bool, device=key_states.device)
    out, scores = attention(
        query_states,
        key_states,
        value_states,
        True, 1 / np.sqrt(HID), mask, 0.9999
    )
    scores = scores.unsqueeze(2)

    return scores, out, num_new_tokens


# H2O KV Cache dropping with Position rolling
class H2OLlamaAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward_old(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if len(past_key_value) > self.layer_idx:
            kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]

        position_ids = torch.arange(kv_seq_len, device=position_ids.device, dtype=position_ids.dtype)
        position_ids = position_ids.unsqueeze(0)

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        # Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids[:, -q_len:])

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        num_new_tokens = None
        if "h2o-linear" in self.config.comment and query_states.size(-2) > 1:
            attn_weights, attn_output, num_new_tokens = attn_batched(query_states, key_states, value_states)
        else:
            attn_weights, attn_output = attn_batched_quad(query_states, key_states, value_states)

        ql, kl = attn_weights.size(-2), attn_weights.size(-1)
        past_key_value_for_layer = self.kv_cache(
            past_key_value[self.layer_idx],
            attn_weights.detach().view(bsz, self.num_key_value_heads, self.num_key_value_groups, ql, kl).amax(dim=2).clone(),
            num_new_tokens=num_new_tokens,
        )

        # # cache returns a tuple, so just insert this tuple as the new cache for this layer
        past_key_value.key_cache[self.layer_idx] = past_key_value_for_layer[0]
        past_key_value.value_cache[self.layer_idx] = past_key_value_for_layer[1]
        if self.layer_idx == 31:
            past_key_value._seen_tokens = past_key_value_for_layer[0].size(-2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # remake causal mask
        pkv_len = 0
        if len(past_key_value) > self.layer_idx:
            pkv_len = past_key_value[self.layer_idx][0].shape[-2]

        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=pkv_len,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]

        if len(past_key_value) > self.layer_idx:
            kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]

        position_ids = torch.arange(kv_seq_len, device=position_ids.device, dtype=position_ids.dtype)
        position_ids = position_ids.unsqueeze(0)

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        # Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids[:, -q_len:])

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        ql, kl = attn_weights.size(-2), attn_weights.size(-1)
        past_key_value_for_layer = self.kv_cache(
            past_key_value[self.layer_idx],
            attn_weights.detach().view(bsz, self.num_key_value_heads, self.num_key_value_groups, ql, kl).amax(dim=2).clone()
        )

        # # cache returns a tuple, so just insert this tuple as the new cache for this layer
        past_key_value.key_cache[self.layer_idx] = past_key_value_for_layer[0]
        past_key_value.value_cache[self.layer_idx] = past_key_value_for_layer[1]
        if self.layer_idx == 31:
            past_key_value._seen_tokens = past_key_value_for_layer[0].size(-2)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# H2O KV Cache dropping with Position rolling
class H2OQwenAttention_streaming(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            raise ValueError("need layer idx")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward_old(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if len(past_key_value) > self.layer_idx:
            kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]

        position_ids = torch.arange(kv_seq_len, device=position_ids.device, dtype=position_ids.dtype)
        position_ids = position_ids.unsqueeze(0)

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single_qwen(query_states, cos, sin, position_ids[:, -q_len:])

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_states = apply_rotary_pos_emb_single_qwen(key_states, cos, sin, position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        num_new_tokens = None
        if "h2o-linear" in self.config.comment and query_states.size(-2) > 1:
            attn_weights, attn_output, num_new_tokens = attn_batched(query_states, key_states, value_states)
        else:
            attn_weights, attn_output = attn_batched_quad(query_states, key_states, value_states)

        ql, kl = attn_weights.size(-2), attn_weights.size(-1)
        past_key_value_for_layer = self.kv_cache(
            past_key_value[self.layer_idx],
            attn_weights.detach().view(bsz, self.num_key_value_heads, self.num_key_value_groups, ql, kl).amax(dim=2).clone(),
            num_new_tokens=num_new_tokens,
        )

        # # cache returns a tuple, so just insert this tuple as the new cache for this layer
        past_key_value.key_cache[self.layer_idx] = past_key_value_for_layer[0]
        past_key_value.value_cache[self.layer_idx] = past_key_value_for_layer[1]
        if self.layer_idx == 27:
            past_key_value._seen_tokens = past_key_value_for_layer[0].size(-2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # remake causal mask
        pkv_len = 0
        if len(past_key_value) > self.layer_idx:
            pkv_len = past_key_value[self.layer_idx][0].shape[-2]

        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=pkv_len,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]

        if len(past_key_value) > self.layer_idx:
            kv_seq_len += past_key_value[self.layer_idx][0].shape[-2]

        position_ids = torch.arange(kv_seq_len, device=position_ids.device, dtype=position_ids.dtype)
        position_ids = position_ids.unsqueeze(0)

        if not position_ids.nelement() > 1:
            position_ids[0][0] = kv_seq_len - 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states = apply_rotary_pos_emb_single_qwen(query_states, cos, sin, position_ids[:, -q_len:])

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_states = apply_rotary_pos_emb_single_qwen(key_states, cos, sin, position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # print(f"{query_states.size()=} {key_states.size()=}")
        s = np.sqrt(np.sqrt(self.head_dim))
        attn_weights = torch.matmul(query_states / s, key_states.transpose(2, 3) / s)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        ql, kl = attn_weights.size(-2), attn_weights.size(-1)
        past_key_value_for_layer = self.kv_cache(
            past_key_value[self.layer_idx],
            attn_weights.detach().view(bsz, self.num_key_value_heads, self.num_key_value_groups, ql, kl).amax(dim=2).clone()
        )

        # # cache returns a tuple, so just insert this tuple as the new cache for this layer
        past_key_value.key_cache[self.layer_idx] = past_key_value_for_layer[0]
        past_key_value.value_cache[self.layer_idx] = past_key_value_for_layer[1]
        if self.layer_idx == 27:
            past_key_value._seen_tokens = past_key_value_for_layer[0].size(-2)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class H2OLlamaForCausalLM_streaming(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention_streaming(config, layer_idx=layer_idx)
            # self.model.layers[layer_idx].self_attn = LlamaAttention(config, layer_idx=layer_idx)


class H2OQwenForCausalLM_streaming(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OQwenAttention_streaming(config, layer_idx=layer_idx)
            # self.model.layers[layer_idx].self_attn = LlamaAttention(config, layer_idx=layer_idx)


def load(model_name_or_path, heavy_hitter=False, args=None):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if args is not None:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        config.hh_size = args.sinks
        config.recent_size = args.window
        config._cascade_stride = args.cascade_stride
        config.comment = args.comment

    if not heavy_hitter:
        raise ValueError("only use this with heavy hitter")

    if "llama" in model_name_or_path.lower():
        model = H2OLlamaForCausalLM_streaming.from_pretrained(
            model_name_or_path,
            device_map="auto",
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    elif "qwen" in model_name_or_path.lower():
        model = H2OQwenForCausalLM_streaming.from_pretrained(
            model_name_or_path,
            device_map="auto",
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    if "h2o-linear" in args.comment:
        print("PATCHING FOR LINEAR H2O")
        model = sample_monkeypatch(model)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer
