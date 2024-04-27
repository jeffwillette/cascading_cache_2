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




if __name__ == "__main__":
    test_compiled()
