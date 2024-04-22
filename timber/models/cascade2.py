import torch
from torch import nn
import os
from typing import List, Optional, Tuple, Dict, Any
import time
import warnings
import numpy as np
from torch.nn import functional as F
from timber.models.cuda_graph import make_graphed_callables


class SinkCache(nn.Module):
    pass


class CascadingSinkCacheCompile(SinkCache):

    def __init__(
        self,
        window_length: int = 8,
        num_sink_tokens: int = 4,
        max_batch_size: int = 1,
        heads: int = 16,
        dim: int = 128,
        n_layers: int = 1,  # need to know in advance for static cache
        max_seq_len: int = 32,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.heads = heads
        self.dim = dim
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype

        self.key_cache: torch.Tensor
        self.score_cache: torch.Tensor
        self.value_cache: torch.Tensor
        self.sink_keys: torch.Tensor
        self.sink_values: torch.Tensor

        self.cascades = max_seq_len // window_length
        self.do_cache = torch.tensor([True for _ in range(self.cascades)],
                                     device=device,
                                     dtype=torch.bool,
                                     requires_grad=False)

        print(f"{self.cascades=} {self.do_cache=}")
        self.do_cache_every_n = torch.tensor(
            [2**i for i in range(self.cascades)],
            device=device,
            dtype=torch.long,
            requires_grad=False,
        )

        self.beta = np.exp(-np.log(100) / window_length)
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = torch.tensor(
            0, device=device, dtype=torch.long, requires_grad=False
        )  # Used in `generate` to keep tally of how many tokens the cache has seen

        self.stored_tokens = torch.tensor([0 for _ in range(self.cascades)],
                                          device=device,
                                          dtype=torch.long,
                                          requires_grad=False)
        self.stored_sinks = torch.tensor(0,
                                         device=device,
                                         dtype=torch.long,
                                         requires_grad=False)

        # each cascade will have start indices which are considered the beginning of
        # the cascade cache to avoid excessive concatenation.
        self.start_indices = torch.tensor([0 for _ in range(self.cascades)],
                                          device=device,
                                          dtype=torch.long,
                                          requires_grad=False)

        # index for positional encodings, this will be modified on
        # each return in order to grab the correct positional encoding indices.
        self.pos = torch.zeros(max_seq_len,
                               device=device,
                               dtype=torch.long,
                               requires_grad=False).view(1, -1)
        self.tmp_arange = torch.arange(self.window_length,
                                       device=device,
                                       dtype=torch.long,
                                       requires_grad=False)
        self.sink_pos = torch.zeros(self.num_sink_tokens,
                                    device=device,
                                    dtype=torch.long,
                                    requires_grad=False).view(1, -1)
        print("INIT NLOGN FAST COMPILED VERSION")

        self.init_static_cache()

    def init_static_cache(self):
        B, H, S, D = self.max_batch_size, self.heads, self.max_seq_len, self.dim
        nsink, dev, dtp = self.num_sink_tokens, self.device, self.dtype

        blank = torch.zeros(B,
                            H,
                            S,
                            D,
                            device=dev,
                            dtype=dtp,
                            requires_grad=False)
        blank_scores = torch.zeros(self.max_seq_len,
                                   device=dev,
                                   dtype=dtp,
                                   requires_grad=False)
        blank_sinks = torch.zeros(B,
                                  H,
                                  nsink,
                                  D,
                                  device=dev,
                                  dtype=dtp,
                                  requires_grad=False)

        self.key_cache = blank.clone()
        self.value_cache = blank.clone()
        self.score_cache = blank_scores.clone()
        self.sink_keys = blank_sinks.clone()
        self.sink_values = blank_sinks.clone()

        self.scatter_idx = torch.ones(1,
                                      device=self.device,
                                      dtype=torch.long,
                                      requires_grad=False)

        self.sink_pos_idx = torch.ones(1,
                                       self.num_sink_tokens,
                                       device=self.device,
                                       dtype=torch.long,
                                       requires_grad=False)

        self.sink_mask = torch.full((1, 1, 1, self.num_sink_tokens),
                                    torch.finfo(self.dtype).min,
                                    device=self.device,
                                    dtype=self.dtype,
                                    requires_grad=False)

        self.mask = torch.full((1, 1, 1, self.max_seq_len),
                               torch.finfo(self.dtype).min,
                               device=self.device,
                               dtype=self.dtype,
                               requires_grad=False)

    def set_cache_bools(self):
        # minus one because seen tokens is incremented before tokens are really added. Therefore we need to subtract that one
        for i, _ in enumerate(self.do_cache):
            if (self._seen_tokens - 1 -
                    self.num_sink_tokens) % self.do_cache_every_n[i] == 0:
                self.do_cache[i] = True
                continue

            self.do_cache[i] = False

    def get_cascade_bounds(self, i):
        # if i == 0:
        #     return self.num_sink_tokens, self.window_length, self.window_length - self.num_sink_tokens
        return self.window_length * i, self.window_length * (
            i + 1), self.window_length

    def get_seq_length(self,
                       layer_idx: Optional[int] = 0,
                       cascade_idx: Optional[int] = -1) -> int:
        return sum([v for v in self.stored_tokens])

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def scat_idx(self, name: str):
        match name:
            case "cache":
                return self.scatter_idx.view(1, 1, 1,
                                             1).repeat(self.max_batch_size,
                                                       self.heads, 1, self.dim)
            case "pos":
                return self.scatter_idx.view(1, 1)
            case "mask":
                return self.scatter_idx.view(1, 1, 1, 1)
            case "score":
                return self.scatter_idx.view(1)

    def add_sinks(
        self,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ):

        cache_idx = self.scat_idx("cache") * stored_sinks
        pos_idx = self.scat_idx("pos") * stored_sinks
        mask_idx = self.scat_idx("mask") * stored_sinks

        sink_keys.scatter_(2, cache_idx, input_key_states)
        sink_values.scatter_(2, cache_idx, input_key_states)
        sink_pos.scatter_(1, pos_idx, stored_sinks.expand_as(pos_idx))
        sink_mask.scatter_(3, mask_idx, 0)

        stored_sinks += 1

        return (
            input_key_states,
            input_value_states,
            input_score_states,
            sink_keys,
            sink_values,
            sink_pos,
            sink_mask,
            keys,
            values,
            pos,
            mask,
            scores,
            stored_sinks,
            start_indices,
            stored_tokens,
        )

    def update_attention_scores(self, scores) -> None:
        self.score_cache = self.beta * self.scores + (1 - self.beta) * scores

    def append_to_cache(
        self,
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ):
        start_idx = torch.gather(start_indices, 0, cascade_idx)
        # start_idx = start_indices[cascade_idx]
        stored = torch.gather(stored_tokens, 0, cascade_idx)
        l, u, segment_len = self.get_cascade_bounds(cascade_idx)
        s = start_idx + l

        # we have empty room in this cache, so we need to shift the index
        # forward by the number of tokens already stored.
        s += stored

        # we do not need to evict, find the end point and insert token
        # since this cache is not full, the insert point will be start + stored_tokens

        cache_idx = self.scat_idx("cache") * s
        score_idx = self.scat_idx("score") * s
        mask_idx = self.scat_idx("mask") * s

        keys.scatter_(2, cache_idx, input_key_states)
        values.scatter_(2, cache_idx, input_value_states)
        scores.scatter_(0, score_idx, input_score_states)
        mask.scatter_(3, mask_idx, 0)

        stored_tokens += F.one_hot(cascade_idx, stored_tokens.size(0))
        return (
            cascade_idx,
            input_key_states,
            input_value_states,
            input_score_states,
            sink_keys,
            sink_values,
            sink_pos,
            sink_mask,
            keys,
            values,
            pos,
            mask,
            scores,
            stored_sinks,
            start_indices,
            stored_tokens,
        )

    def warn(self, args):
        warnings.warn(
            "the cascading cache is full, evicted context from the last cascade will be dropped"
        )
        return args

    def overwrite_cache(
        self,
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ):
        l, _, _ = self.get_cascade_bounds(cascade_idx)
        start_idx = torch.gather(start_indices, 0, cascade_idx)
        s = start_idx + l

        cache_idx = self.scat_idx("cache") * s
        score_idx = self.scat_idx("score") * s

        keys.scatter_(2, cache_idx, input_key_states)
        values.scatter_(2, cache_idx, input_value_states)
        scores.scatter_(0, score_idx, input_score_states)

    def evict_from_cache(
        self,
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ):
        start_idx = torch.gather(start_indices, 0, cascade_idx)
        l, u, segment_len = self.get_cascade_bounds(cascade_idx)
        s = start_idx + l

        # we need to evict
        # 1. find the oldest token (start point), remove it and
        #    set input_key_state at that location

        cache_idx = self.scat_idx("cache") * s
        score_idx = self.scat_idx("score") * s

        next_input_key_state = torch.gather(keys, 2, cache_idx).clone()
        next_input_value_state = torch.gather(values, 2, cache_idx).clone()
        next_input_score_state = torch.gather(scores, 0, score_idx).clone()

        keys.scatter_(2, cache_idx, input_key_states)
        values.scatter_(2, cache_idx, input_value_states)
        scores.scatter_(0, score_idx, input_score_states)

        # 2. rotate the start index.
        # new_start_idx = (start_idx + 1) % segment_len (vectorized version of this)
        start_indices = (start_indices + F.one_hot(
            cascade_idx, start_indices.size(0))) % segment_len

        # mask remains unchanged for this operation.
        return (
            cascade_idx,
            next_input_key_state,
            next_input_value_state,
            next_input_score_state,
            sink_keys,
            sink_values,
            sink_pos,
            sink_mask,
            keys,
            values,
            pos,
            mask,
            scores,
            stored_sinks,
            start_indices,
            stored_tokens,
        )

    def add_keys(
        self,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ):

        for i in range(self.cascades):
            l, u, segment_len = self.get_cascade_bounds(i)
            cascade_idx = torch.tensor(i, device=self.device, dtype=torch.long)

            o = (
                cascade_idx,
                input_key_states,
                input_value_states,
                input_score_states,
                sink_keys,
                sink_values,
                sink_pos,
                sink_mask,
                keys,
                values,
                pos,
                mask,
                scores,
                stored_sinks,
                start_indices,
                stored_tokens,
            )

            if self.do_cache[i]:
                if stored_tokens[i] < segment_len:
                    (
                        _,
                        input_key_states,
                        input_value_states,
                        input_score_states,
                        sink_keys,
                        sink_values,
                        sink_pos,
                        sink_mask,
                        keys,
                        values,
                        pos,
                        mask,
                        scores,
                        stored_sinks,
                        start_indices,
                        stored_tokens,
                    ) = self.append_to_cache(*o)
                    break
                else:
                    (
                        _,
                        input_key_states,
                        input_value_states,
                        input_score_states,
                        sink_keys,
                        sink_values,
                        sink_pos,
                        sink_mask,
                        keys,
                        values,
                        pos,
                        mask,
                        scores,
                        stored_sinks,
                        start_indices,
                        stored_tokens,
                    ) = self.evict_from_cache(*o)
                    # since we evicted a token, we need to move it along for the
                    # next cascade layer to deal with recursively (if there is a next cascade layer)
                    if i + 1 > (self.cascades - 1):
                        break
            else:
                if stored_tokens[i] == 0:
                    # if we are not supposed to move the cache, but we were called
                    # with states as an input. Then there are two possibilities:
                    # 1. We are not supposed to do cache, but the length of this cache is zero.
                    #    this may happen due to the do_cache input_values not lining up perfectly with powers of 2.
                    #    In this case, we should add an element to the cache so it doesn't just get automatically evicted.
                    (
                        _,
                        input_key_states,
                        input_value_states,
                        input_score_states,
                        sink_keys,
                        sink_values,
                        sink_pos,
                        sink_mask,
                        keys,
                        values,
                        pos,
                        mask,
                        scores,
                        stored_sinks,
                        start_indices,
                        stored_tokens,
                    ) = self.append_to_cache(*o)
                    break
                else:
                    # 2. Since we know this cache has something in it, and we are not to do caching,
                    #    find the oldest thing in this cache, compare attention input_scores,
                    #    and remove if needed.

                    s = start_indices[i] + l

                    score_idx = self.scat_idx("score") * s
                    old_input_score = torch.gather(scores, 0,
                                                   score_idx) / (1 - self.beta)
                    if old_input_score >= input_score_states / (1 - self.beta):
                        # old input_score is better, do nothing.
                        # break onstead of cotinue because this stops the cascade
                        break

                    (
                        _,
                        input_key_states,
                        input_value_states,
                        input_score_states,
                        sink_keys,
                        sink_values,
                        sink_pos,
                        sink_mask,
                        keys,
                        values,
                        pos,
                        mask,
                        scores,
                        stored_sinks,
                        start_indices,
                        stored_tokens,
                    ) = self.overwrite_cache(*o)

        pos_ub = stored_tokens.sum()  # same as get_Seq_len

        for i in range(self.cascades):
            if stored_tokens[i] == 0:
                break

            l, u, seg_len = self.get_cascade_bounds(i)
            u = min(u, l + stored_tokens[i])
            seg_len = min(stored_tokens[i], seg_len)
            start_idx = start_indices[i]

            pos[0, l:u] = (self.tmp_arange[:pos_ub] +
                           (seg_len - start_idx)) % seg_len
            pos[0, l:u] += pos_ub - seg_len
            pos_ub = pos_ub - seg_len

        return (
            input_key_states,
            input_value_states,
            input_score_states,
            sink_keys,
            sink_values,
            sink_pos,
            sink_mask,
            keys,
            values,
            pos,
            mask,
            scores,
            stored_sinks,
            start_indices,
            stored_tokens,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        create_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self._seen_tokens += key_states.shape[-2]
        self.set_cache_bools()

        score_states = torch.zeros(1,
                                   device=key_states.device,
                                   dtype=key_states.dtype,
                                   requires_grad=False)

        o = (
            key_states,
            value_states,
            score_states,
            self.sink_keys,
            self.sink_values,
            self.sink_pos,
            self.sink_mask,
            self.key_cache,
            self.value_cache,
            self.pos,
            self.mask,
            self.score_cache,
            self.stored_sinks,
            self.start_indices,
            self.stored_tokens,
        )

        if self.stored_sinks < self.num_sink_tokens:
            o = self.add_sinks(*o)
        else:
            o = self.add_keys(*o)

        (
            _,
            _,
            _,
            sink_keys,
            sink_values,
            sink_pos,
            sink_mask,
            keys,
            values,
            pos,
            mask,
            scores,
            stored_sinks,
            start_indices,
            stored_tokens,
        ) = o
        # print(f"\n\n\nbefore")
        # print(
        #     f"{self.sink_keys=}\n{self.sink_values=}\n{self.sink_pos=}\n{self.sink_mask=}"
        # )

        self.sink_keys = sink_keys
        self.sink_values = sink_values
        self.sink_pos = sink_pos
        self.sink_mask = sink_mask
        self.key_cache = keys
        self.value_cache = values
        self.pos = pos
        self.mask = mask
        self.score_cache = scores
        self.stored_sinks = stored_sinks
        self.start_indices = start_indices
        self.stored_tokens = stored_tokens

        # print(f"\n\n\nafter")
        # print(
        #     f"{self.sink_keys=}\n{self.sink_values=}\n{self.sink_pos=}\n{self.sink_mask=}"
        # )

        return (
            self.sink_keys,
            self.sink_values,
            self.sink_pos,
            self.sink_mask,
            self.key_cache,
            self.value_cache,
            self.pos + self.num_sink_tokens,
            self.mask,
        )


class CascadingSinkCacheSlow(SinkCache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[List[torch.Tensor]] = [[]]
        self.attn_score_cache: List[List[torch.Tensor]] = [[]]
        self.value_cache: List[List[torch.Tensor]] = [[]]

        # self.cache_lens = [window_length // 2**i for i in range(cascades)]
        print("INIT NLOGN VERSION")

        self.sink_keys: List[torch.Tensor] = []
        self.sink_values: List[torch.Tensor] = []

        # self.do_cache = [True for _ in range(cascades)]
        # self.do_cache_every_n = [2**i for i in range(cascades)]
        self.do_cache = [True]
        self.do_cache_every_n = [1]
        self.beta = np.exp(-np.log(100) / window_length)
        # self.do_cache_every_n = [1, 2**7]
        # self.do_cache_every_n = [1, 2**4, 2**8, 2**12]
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def set_cache_bools(self):
        for i, _ in enumerate(self.do_cache):
            if (self._seen_tokens - 1 -
                    self.num_sink_tokens) % self.do_cache_every_n[i] == 0:
                self.do_cache[i] = True
                continue
            self.do_cache[i] = False

    def get_seq_length(self,
                       layer_idx: Optional[int] = 0,
                       cascade_idx: Optional[int] = -1) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length

        if cascade_idx == -1:
            # get total length
            return sum([
                self.key_cache[i][layer_idx].size(-2)
                if len(self.key_cache[i]) > 0 else 0
                for i, _ in enumerate(self.key_cache)
            ])

        if len(self.key_cache[cascade_idx]) <= layer_idx:
            return 0
        return self.key_cache[cascade_idx][layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update_attention_scores(
        self,
        attention_scores: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        if self._seen_tokens <= self.num_sink_tokens:
            return

        start, end = self.num_sink_tokens, self.num_sink_tokens
        for i, _ in enumerate(self.key_cache):
            start = end
            end = end + self.key_cache[i][layer_idx].size(-2)

            if attention_scores.size(-1) <= start or len(
                    self.attn_score_cache[i]) <= layer_idx:
                return

            chunk_attn_scores = attention_scores[:, start:end]

            # self.attn_score_cache[i][layer_idx] += chunk_attn_scores
            self.attn_score_cache[i][
                layer_idx] = self.beta * self.attn_score_cache[i][
                    layer_idx] + (1 - self.beta) * chunk_attn_scores

    def print_cache_size(self, layer_idx):
        for i, _ in enumerate(self.key_cache):
            # [bsz, num_heads, seq_len, head_dim]

            if len(self.key_cache[i]) <= layer_idx:
                print(f"need to append cache for cascade {i}")
                continue

            key_cache = self.key_cache[i][layer_idx]
            print(f"key cache size: {key_cache.size()=} for cascade: {i}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        # Update the number of seen tokens

        # self.print_cache_size(layer_idx)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            self.set_cache_bools()

        # print(f"{[len(v) for v in self.key_cache]=}")
        # print(f"{self.do_cache=} {self._seen_tokens=}")
        if len(self.sink_keys) <= layer_idx:
            self.sink_keys.append(key_states)
            self.sink_values.append(value_states)
            # print(f"first sink keys {layer_idx=} ", self.sink_keys[layer_idx].size())
            return self.sink_keys[layer_idx], self.sink_values[layer_idx]
        elif self.sink_keys[layer_idx].size(-2) < self.num_sink_tokens:
            self.sink_keys[layer_idx] = torch.cat(
                (self.sink_keys[layer_idx], key_states), dim=-2)
            self.sink_values[layer_idx] = torch.cat(
                (self.sink_values[layer_idx], value_states), dim=-2)
            # print(f"second sink keys {layer_idx=} ", self.sink_keys[layer_idx].size())
            return self.sink_keys[layer_idx], self.sink_values[layer_idx]

        # print(f"after sink keys {layer_idx=} ",
        #       self.sink_keys[layer_idx].size())

        # we don't know the score states yet, but just make a placeholder for the new scores so they
        # can be treated like everything else.
        score_states = torch.zeros(key_states.size(0),
                                   key_states.size(-2),
                                   device=key_states.device,
                                   dtype=key_states.dtype)

        for i, _ in enumerate(self.key_cache):
            # [bsz, num_heads, seq_len, head_dim]

            # print(
            #     f"iteration: {i} {len(self.key_cache[i])=} {layer_idx=} {key_states.shape[-2]=}"
            # )

            if len(self.key_cache[i]) <= layer_idx:
                # Empty cache
                self.key_cache[i].append(key_states)
                self.value_cache[i].append(value_states)
                self.attn_score_cache[i].append(score_states.clone())

                # print(f"if block: {self.key_cache[i][layer_idx].size()=}")
                break

            elif key_states.shape[-2] + self.get_seq_length(
                    layer_idx, i) <= self.get_max_length():
                # Growing cache
                if not self.do_cache[i]:
                    # we have the evcited tokens from the last cascade layer,
                    # so compare them to the last tokens in this layer and keep the
                    # ones with a larger total attention score.
                    prev_score = self.attn_score_cache[i][layer_idx][:, -1:]
                    # print(f"{prev_score.size()=} {score_states.size()=}")
                    if prev_score[0, 0] / (1 - self.beta) >= score_states[
                            0, 0] / (1 - self.beta):
                        break

                    self.attn_score_cache[i][layer_idx][:, -1:] = score_states
                    self.key_cache[i][layer_idx][:, :, -1:] = key_states
                    self.value_cache[i][layer_idx][:, :, -1:] = value_states
                    # print(
                    #     f"elif block not do cache: {self.key_cache[i][layer_idx].size()=}"
                    # )
                    break

                self.key_cache[i][layer_idx] = torch.cat(
                    [self.key_cache[i][layer_idx], key_states], dim=-2)

                self.value_cache[i][layer_idx] = torch.cat(
                    [self.value_cache[i][layer_idx], value_states], dim=-2)

                self.attn_score_cache[i][layer_idx] = torch.cat(
                    (self.attn_score_cache[i][layer_idx],
                     score_states.clone()),
                    dim=-1)
                # print(f"elif block: {self.key_cache[i][layer_idx].size()=}")
                break

            else:
                if not self.do_cache[i]:
                    # we have the evcited tokens from the last cascade layer,
                    # so compare them to the last tokens in this layer and keep the
                    # ones with a larger total attention score.
                    prev_score = self.attn_score_cache[i][layer_idx][:, -1:]
                    # print(f"{prev_score.size()=} {score_states.size()=}")
                    if prev_score[0, 0] / (1 - self.beta) >= score_states[
                            0, 0] / (1 - self.beta):
                        break

                    self.attn_score_cache[i][layer_idx][:, -1:] = score_states
                    self.key_cache[i][layer_idx][:, :, -1:] = key_states
                    self.value_cache[i][layer_idx][:, :, -1:] = value_states
                    break

                key_cache = self.key_cache[i][layer_idx]
                value_cache = self.value_cache[i][layer_idx]
                score_cache = self.attn_score_cache[i][layer_idx]

                # Shifting cache
                keys_to_keep = key_cache[:, :, -self.get_max_length() +
                                         key_states.shape[-2]:]

                scores_to_keep = score_cache[:, -self.get_max_length() +
                                             key_states.shape[-2]:]

                values_to_keep = value_cache[:, :, -self.get_max_length() +
                                             value_states.shape[-2]:]

                # print(f"catting cache: {i}")
                self.key_cache[i][layer_idx] = torch.cat(
                    [keys_to_keep, key_states], dim=-2)
                self.value_cache[i][layer_idx] = torch.cat(
                    [values_to_keep, value_states], dim=-2)
                self.attn_score_cache[i][layer_idx] = torch.cat(
                    (scores_to_keep, score_states.clone()), dim=-1)

                # these are the evicted tokens for the next iteration
                key_states = key_cache[:, :, :-self.get_max_length() +
                                       key_states.shape[-2]]
                score_states = score_cache[:, :-self.get_max_length() +
                                           key_states.shape[-2]]
                value_states = value_cache[:, :, :-self.get_max_length() +
                                           value_states.shape[-2]]

                # if true, we have evicted tokens, but nowhere to put them. In this case,
                # we need to add another cascade to the key cache.
                if i == len(self.key_cache) - 1:
                    self.key_cache.append([])
                    self.value_cache.append([])
                    self.attn_score_cache.append([])

                    self.key_cache[i + 1].append(key_states)
                    self.value_cache[i + 1].append(value_states)
                    self.attn_score_cache[i + 1].append(score_states)

                    self.do_cache += [None]
                    self.do_cache_every_n += [2**(i + 1)]
                    self.set_cache_bools()
                    # print(
                    #     f"ADDED CACHE {self.key_cache[i + 1][layer_idx].size()=}"
                    # )
                    break
                # print(
                #     f"else block: {self.key_cache[i][layer_idx].size()=} {i=} {layer_idx=}"
                # )
                # print(f"{key_states.size()=}")

        # if self.seen_tokens % 1 == 0:
        #     self.print_cache_size(layer_idx)
        #     print(f"{self.do_cache=} {self.do_cache_every_n=}")

        # print(f"doing out for layer: {layer_idx=}")
        out_keys, out_values = [], []
        for i in reversed(range(len(self.key_cache))):
            # print(f"{i=} {len(self.key_cache[i])=}")
            # print(f"{len(self.key_cache[i])=} {len(self.value_cache[i])=}")
            if len(self.key_cache[i]) > 0 and len(self.value_cache[i]) > 0:
                out_keys += [self.key_cache[i][layer_idx]]
                out_values += [self.value_cache[i][layer_idx]]

        # for k, v in zip(out_keys, out_values):
        #     print(f"out sizes: {k.size()=} {v.size()=}")

        out_keys = torch.cat([self.sink_keys[layer_idx]] + out_keys, dim=-2)
        out_values = torch.cat([self.sink_values[layer_idx]] + out_values,
                               dim=-2)

        return out_keys, out_values


class StaticSinkCache(SinkCache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length + self.num_sink_tokens

    def update_orig(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(
                layer_idx) < self.get_max_length():
            # Growing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][:, :,
                                                     -self.window_length +
                                                     key_states.shape[-2]:]

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, :self.num_sink_tokens]

            self.key_cache[layer_idx] = torch.cat(
                [sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, :self.
                                                      num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][:, :,
                                                         -self.window_length +
                                                         value_states.
                                                         shape[-2]:]

            self.value_cache[layer_idx] = torch.cat(
                [sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

            if not hasattr(self, "keeps"):
                self.keeps = []

            keep = torch.rand(1).item() > 0.99
            if keep:
                for i in range(len(self.keeps)):
                    self.keeps[i].append(self._seen_tokens +
                                         i * self.window_length)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.keeps.append([])

        elif key_states.shape[-2] + self.get_seq_length(
                layer_idx) < self.get_max_length():
            # Growing cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, :self.num_sink_tokens]

            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][:, :,
                                                     -self.window_length +
                                                     key_states.shape[-2]:]

            sink_values = self.value_cache[layer_idx][:, :, :self.
                                                      num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][:, :,
                                                         -self.window_length +
                                                         value_states.
                                                         shape[-2]:]

            evicted_keys = self.key_cache[
                layer_idx][:, :, self.num_sink_tokens:-self.window_length +
                           key_states.shape[-2]]

            evicted_values = self.value_cache[
                layer_idx][:, :, self.num_sink_tokens:-self.window_length +
                           value_states.shape[-2]]

            if self._seen_tokens not in self.keeps[layer_idx]:
                evicted_keys = evicted_keys[:, :, :-1]
                evicted_values = evicted_values[:, :, :-1]

            self.key_cache[layer_idx] = torch.cat(
                [sink_keys, evicted_keys, keys_to_keep, key_states], dim=-2)

            self.value_cache[layer_idx] = torch.cat(
                [sink_values, evicted_values, values_to_keep, value_states],
                dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[
                layer_idx].index_select(0, beam_idx.to(device))


def test_compiled():
    window, sink = 512, 4
    dim, head, layers = 128, 16, 1
    max_seq = 4096
    device = "cuda:0"
    dtype = torch.float16
    cache = CascadingSinkCacheCompile(
        window_length=window,
        num_sink_tokens=sink,
        max_batch_size=1,
        heads=head,
        dim=dim,
        n_layers=layers,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq,
    )

    cascade_idx = torch.tensor(0,
                               device=device,
                               dtype=torch.long,
                               requires_grad=False)
    input_key_states = torch.randn(1,
                                   head,
                                   1,
                                   dim,
                                   device=device,
                                   dtype=dtype,
                                   requires_grad=False)
    input_value_states = torch.randn(1,
                                     head,
                                     1,
                                     dim,
                                     device=device,
                                     dtype=dtype,
                                     requires_grad=False)
    input_score_states = torch.randn(1,
                                     device=device,
                                     dtype=dtype,
                                     requires_grad=False)
    sink_keys = torch.randn(1,
                            head,
                            sink,
                            dim,
                            device=device,
                            dtype=dtype,
                            requires_grad=False)
    sink_values = torch.randn(1,
                              head,
                              sink,
                              dim,
                              device=device,
                              dtype=dtype,
                              requires_grad=False)
    sink_pos = torch.arange(sink,
                            device=device,
                            dtype=torch.long,
                            requires_grad=False).view(1, -1)
    sink_mask = torch.zeros(1,
                            1,
                            1,
                            sink,
                            device=device,
                            dtype=dtype,
                            requires_grad=False)
    keys = torch.randn(1,
                       head,
                       max_seq,
                       dim,
                       device=device,
                       dtype=dtype,
                       requires_grad=False)
    values = torch.randn(1,
                         head,
                         max_seq,
                         dim,
                         device=device,
                         dtype=dtype,
                         requires_grad=False)
    pos = torch.zeros(max_seq,
                      device=device,
                      dtype=torch.long,
                      requires_grad=False).view(1, -1)
    mask = torch.zeros(1,
                       1,
                       1,
                       max_seq,
                       device=device,
                       dtype=dtype,
                       requires_grad=False)
    scores = torch.rand(max_seq,
                        device=device,
                        dtype=dtype,
                        requires_grad=False)
    stored_sinks = torch.zeros(1,
                               device=device,
                               dtype=torch.long,
                               requires_grad=False)
    start_indices = torch.zeros(max_seq // window,
                                device=device,
                                dtype=torch.long,
                                requires_grad=False)

    stored_tokens = torch.zeros(max_seq // window,
                                device=device,
                                dtype=torch.long,
                                requires_grad=False)

    cache.eval()

    cache.append_to_cache = make_graphed_callables(cache.append_to_cache, (
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ),
                                                   allow_unused_input=True)

    cache.evict_from_cache = make_graphed_callables(cache.evict_from_cache, (
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ),
                                                    allow_unused_input=True)

    cache.overwrite_cache = make_graphed_callables(cache.overwrite_cache, (
        cascade_idx,
        input_key_states,
        input_value_states,
        input_score_states,
        sink_keys,
        sink_values,
        sink_pos,
        sink_mask,
        keys,
        values,
        pos,
        mask,
        scores,
        stored_sinks,
        start_indices,
        stored_tokens,
    ),
                                                   allow_unused_input=True)

    # cache = torch.compile(cache, mode="reduce-overhead", fullgraph=True)
    slow_cache = CascadingSinkCacheSlow(window_length=window,
                                        num_sink_tokens=sink)

    with torch.no_grad():
        slow_times, fast_times = [], []
        for i in range(6000):
            for layer_idx in range(layers):
                # print(f"{'='*50}")
                if i < sink:
                    k, v = torch.ones(
                        1, head, 1, dim, device=device,
                        dtype=dtype) * (i + 1), torch.ones(
                            1, head, 1, dim, device=device,
                            dtype=dtype) * (i + 1)
                else:
                    k, v = torch.ones(
                        1, head, 1, dim, device=device,
                        dtype=dtype) * (i + 1), torch.ones(
                            1, head, 1, dim, device=device,
                            dtype=dtype) * (i + 1)

                # print(f"\n\n\n\ninput for {layer_idx=} kv {k.size()=} {v.size()=}")

                tic = time.perf_counter()
                slow_k, slow_v = slow_cache.update(k.clone(),
                                                   v.clone(),
                                                   layer_idx=layer_idx)
                slow_times.append(time.perf_counter() - tic)

                tic = time.perf_counter()
                k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask = cache.update(
                    k.clone(), v.clone(), layer_idx=layer_idx)
                fast_times.append(time.perf_counter() - tic)

                # print(f"\n\n{k.size()=} {k_nosink.size()=}")
                k, v = torch.cat((k, k_nosink), dim=-2), torch.cat(
                    (v, v_nosink), dim=-2)
                pos = torch.cat((pos, pos_nosink), dim=-1).squeeze(0)
                mask = torch.cat((sink_mask, mask), dim=-1)

                n = (mask == 0).sum()
                k, v = k[:, :, :n], v[:, :, :n]
                argsort = torch.argsort(pos[:n])
                # print(f"{mask=} {k=} {v=}")

                # print(
                #     f"before sort: \n{slow_k.view(-1)=}\n{k.reshape(-1)=}\n{pos.reshape(-1)=}"
                # )
                # print(f"{pos=}")

                k, v = k[:, :, argsort], v[:, :, argsort]

                # print(f"after sort: {k.view(-1)}")
                if not slow_k.size() == k.size():
                    print(f"{slow_k.size()=} {k.size()=}")
                    print(f"sizes not equal...\n{slow_k=} {k=}")
                    exit()

                diff = (slow_k - k).abs().amax()
                print(f"k diff: {diff=} {i=} {layer_idx=}")
                if diff > 1e-6:
                    print(
                        f"{slow_k.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - slow_k).abs().view(-1)=}"
                    )
                    exit("too big")

                # print(
                #     f"output for {layer_idx=} {sk.size()=} {sv.size()=} {spos.size()=}"
                # )
    slow_times = sum(slow_times) / len(slow_times)
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{slow_times=} {fast_times=}")


if __name__ == "__main__":
    test_compiled()
