import torch
from torch import nn
import time
from typing import List, Optional, Tuple, Dict, Any
import warnings
import numpy as np
from torch.nn import functional as F
from timber.models.cuda_graph import make_graphed_callables


class SinkCache(nn.Module):
    pass


class CascadingSinkCache(SinkCache):

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

        self.key_cache: List[torch.Tensor] = []
        self.score_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.sink_keys: List[torch.Tensor] = []
        self.sink_values: List[torch.Tensor] = []

        self.cascades = max_seq_len // window_length
        self.do_cache = [True for _ in range(self.cascades)]
        self.do_cache_every_n = [2**i for i in range(self.cascades)]
        # self.do_cache_every_n = [i + 1 for i in range(self.cascades)]

        self.beta = np.exp(-np.log(100) / window_length)
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        # per layer, not per cascade
        self._stored_tokens = [[0 for _ in range(n_layers)]
                               for _ in range(self.cascades)]
        self._stored_sinks = [0 for _ in range(n_layers)]

        # each cascade will have start indices which are considered the beginning of
        # the cascade cache to avoid excessive concatenation.
        self.start_indices = [[0 for _ in range(n_layers)]
                              for _ in range(self.cascades)]

        # index for positional encodings, this will be modified on
        # each return in order to grab the correct positional encoding indices.
        self.pos_idx = torch.arange(max_seq_len,
                                    device=device,
                                    dtype=torch.long)
        self.tmp_arange = torch.arange(self.window_length,
                                       device=device,
                                       dtype=torch.long)
        self.sink_pos = torch.arange(self.num_sink_tokens,
                                     device=device,
                                     dtype=torch.long)
        # print("INIT NLOGN FAST VERSION")

        self.init_static_cache()

    def init_static_cache(self):
        B, H, S, D = self.max_batch_size, self.heads, self.max_seq_len, self.dim
        nsink, dev, dtp = self.num_sink_tokens, self.device, self.dtype

        blank = torch.zeros(B, H, S, D, device=dev, dtype=dtp)
        blank_scores = torch.zeros(self.max_seq_len, device=dev, dtype=dtp)
        blank_sinks = torch.zeros(B, H, nsink, D, device=dev, dtype=dtp)

        for i in range(self.n_layers):
            if len(self.key_cache) <= i:
                self.key_cache.append(blank.clone())
                self.value_cache.append(blank.clone())
                self.score_cache.append(blank_scores.clone())
                self.sink_keys.append(blank_sinks.clone())
                self.sink_values.append(blank_sinks.clone())

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
        return sum([v[layer_idx] for v in self._stored_tokens])

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def get_cache(self, cache: str, layer_idx: int, frm: int, to: int):
        match cache:
            case "key":
                return self.key_cache[layer_idx][:, :, frm:to]
            case "value":
                return self.value_cache[layer_idx][:, :, frm:to]
            case "score":
                return self.score_cache[layer_idx][frm:to]
            case "sink_keys":
                return self.sink_keys[layer_idx][:, :, frm:to]
            case "sink_values":
                return self.sink_values[layer_idx][:, :, frm:to]
            case _:
                raise NotImplementedError(f"got unkknown cache name: {cache=}")

    def set_cache(self, cache: str, layer_idx: int, frm: int, to: int,
                  val: torch.Tensor):
        match cache:
            case "key":
                self.key_cache[layer_idx][:, :, frm:to] = val
            case "value":
                self.value_cache[layer_idx][:, :, frm:to] = val
            case "score":
                self.score_cache[layer_idx][frm:to] = val
            case "sink_keys":
                self.sink_keys[layer_idx][:, :, frm:to] = val
            case "sink_values":
                self.sink_values[layer_idx][:, :, frm:to] = val
            case _:
                raise NotImplementedError(f"got unkknown cache name: {cache=}")

    def add_sinks(self, key_states, value_states, layer_idx, create_mask):
        stored = self._stored_sinks[layer_idx]
        self.set_cache("sink_keys", layer_idx, stored, stored + 1, key_states)
        self.set_cache("sink_values", layer_idx, stored, stored + 1,
                       value_states)
        self._stored_sinks[layer_idx] += 1

        sink_mask, mask = None, None
        keys, values = None, None
        pos_idx = None

        if create_mask:
            sink_keys = self.get_cache("sink_keys", layer_idx, 0,
                                       self.num_sink_tokens)
            sink_values = self.get_cache("sink_values", layer_idx, 0,
                                         self.num_sink_tokens)
            sink_mask = torch.zeros(self.num_sink_tokens,
                                    device=self.device,
                                    dtype=self.dtype)

            end = self._stored_sinks[layer_idx]
            sink_mask[end:] = torch.finfo(self.dtype).min

            mask = torch.full((self.max_batch_size, 1, 1, self.max_seq_len),
                              torch.finfo(self.dtype).min,
                              device=self.device,
                              dtype=self.dtype)

            sink_mask = sink_mask.view(self.max_batch_size, 1, 1,
                                       self.num_sink_tokens)
            sink_pos = self.sink_pos

            pos_idx = self.pos_idx.unsqueeze(0)
            pos_idx[:, :] = 0

            keys = self.get_cache("key", layer_idx, 0, self.max_seq_len)
            values = self.get_cache("value", layer_idx, 0, self.max_seq_len)

        else:
            sink_keys = self.get_cache("sink_keys", layer_idx, 0, stored + 1)
            sink_values = self.get_cache("sink_values", layer_idx, 0,
                                         stored + 1)
            sink_pos = self.sink_pos[:stored + 1]

        return (
            sink_keys,
            sink_values,
            sink_pos.unsqueeze(0),
            sink_mask,
            keys,
            values,
            pos_idx,
            mask,
        )

    def update_attention_scores(
        self,
        attention_scores: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        end = attention_scores.size(0)
        old_scores = self.get_cache('score', layer_idx, 0, end)
        self.set_cache(
            "score", layer_idx, 0, end,
            self.beta * old_scores + (1 - self.beta) * attention_scores)
        # print(
        #     f"score states after update: {self.get_cache('score', layer_idx, 0, end)}"
        # )

    def update_segment(self, layer_idx, cascade_idx, key_states, value_states,
                       score_states):

        # print(f"called update_segment for {cascade_idx=}")
        start_idx = self.start_indices[cascade_idx][layer_idx]
        stored = self._stored_tokens[cascade_idx][layer_idx]
        l, u, segment_len = self.get_cascade_bounds(cascade_idx)

        # print(f"{self.do_cache=}")
        if self.do_cache[cascade_idx]:
            # if we are supposed to do the cache, then we need to check if we need
            # to evict one token or not.

            s = start_idx + l
            if self._stored_tokens[cascade_idx][layer_idx] < segment_len:
                # we have empty room in this cache, so we need to shift the index
                # forward by the number of tokens already stored.
                s += stored
                # we do not need to evict, find the end point and insert token
                # since this cache is not full, the insert point will be start + stored_tokens
                self.set_cache("key", layer_idx, s, s + 1, key_states)
                self.set_cache("value", layer_idx, s, s + 1, value_states)
                self.set_cache("score", layer_idx, s, s + 1, score_states)
                self._stored_tokens[cascade_idx][layer_idx] += 1
                return

            else:
                # we need to evict
                # 1. find the oldest token (start point), remove it and
                #    set key_state at that location
                next_key_state = self.get_cache("key", layer_idx, s,
                                                s + 1).clone()
                next_value_state = self.get_cache("value", layer_idx, s,
                                                  s + 1).clone()
                next_score_state = self.get_cache("score", layer_idx, s,
                                                  s + 1).clone()

                self.set_cache("key", layer_idx, s, s + 1, key_states)
                self.set_cache("value", layer_idx, s, s + 1, value_states)
                self.set_cache("score", layer_idx, s, s + 1, score_states)

                # 2. increment the start point by one, being mindful of
                #    the upper bound.
                new_start_idx = (start_idx + 1) % segment_len
                self.start_indices[cascade_idx][layer_idx] = new_start_idx

                # 3. since we evicted a token, we need to move it along for the
                #    next cascade layer to deal with recursively (if there is a next cascade layer)
                if cascade_idx + 1 <= self.cascades - 1:
                    return self.update_segment(
                        layer_idx,
                        cascade_idx + 1,
                        next_key_state,
                        next_value_state,
                        next_score_state,
                    )

                warnings.warn(
                    "the cascading cache is full, evicted context from the last cascade will be dropped"
                )
                return

        else:
            # if we are not supposed to move the cache, but we were called
            # with states as an input. Then there are two possibilities:
            # 1. We are not supposed to do cache, but the length of this cache is zero.
            #    this may happen due to the do_cache values not lining up perfectly with powers of 2.
            #    In this case, we should add an element to the cache so it doesn't just get automatically evicted.
            s = start_idx + l
            if self._stored_tokens[cascade_idx][layer_idx] == 0:
                s += stored
                # we do not need to evict, find the end point and insert token
                # since this cache is not full, the insert point will be start + stored_tokens
                self.set_cache("key", layer_idx, s, s + 1, key_states)
                self.set_cache("value", layer_idx, s, s + 1, value_states)
                self.set_cache("score", layer_idx, s, s + 1, score_states)
                self._stored_tokens[cascade_idx][layer_idx] += 1
                return

            # 2. Since we know every cache has something in it, find the newest (most recent) thing
            #    in this cache, compare attention scores,
            #    and remove if needed.
            # oldest_idx = (start_idx - 1) % segment_len

            s = (start_idx - 1) % self._stored_tokens[cascade_idx][layer_idx]
            s = s + l

            old_score = self.get_cache("score", layer_idx, s, s + 1)
            if old_score.item() / (1 - self.beta) > score_states.item() / (
                    1 - self.beta):
                return  # old score is better, do nothing

            self.set_cache("key", layer_idx, s, s + 1, key_states)
            self.set_cache("value", layer_idx, s, s + 1, value_states)
            self.set_cache("score", layer_idx, s, s + 1, score_states)
            return

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        create_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        score_states = torch.zeros(1,
                                   device=key_states.device,
                                   dtype=key_states.dtype)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            self.set_cache_bools()

        if self._stored_sinks[layer_idx] < self.num_sink_tokens:
            return self.add_sinks(key_states, value_states, layer_idx,
                                  create_mask)

        self.update_segment(layer_idx, 0, key_states, value_states,
                            score_states)

        end = self.get_seq_length(layer_idx)
        key_states = self.get_cache("key", layer_idx, 0, self.get_max_length())
        value_states = self.get_cache("value", layer_idx, 0,
                                      self.get_max_length())
        # if layer_idx == 0:
        #     print(f"before updating: {self.pos_idx[:end]=}")

        pos_ub = end
        # print(
        #     f"{pos_ub=}\n{self.pos_idx=}\n{self.start_indices=}\n{self._stored_tokens=}"
        # )
        pos_idx = self.pos_idx[:pos_ub]
        for i in range(self.cascades):
            if self._stored_tokens[i][layer_idx] == 0:
                break

            l, u, seg_len = self.get_cascade_bounds(i)
            # print(f"before minimum {u=} {seg_len=}")
            u = min(u, l + self._stored_tokens[i][layer_idx])
            seg_len = min(self._stored_tokens[i][layer_idx], seg_len)
            start_idx = self.start_indices[i][layer_idx]

            # print(f"lower bound: {l} upper bound: {u} {seg_len=} {start_idx=}")
            # print(f"{self.tmp_arange=}")

            # print(
            #     f"befor emodulo {pos_idx[l:u]=} {self.tmp_arange[:pos_ub]=} {start_idx=} {seg_len=}"
            # )
            # rotate the

            # if self.do_cache[i]:
            # pos_idx[l:u] = (self.tmp_arange[:pos_ub] + start_idx) % seg_len
            pos_idx[l:u] = (self.tmp_arange[:pos_ub] +
                            (seg_len - start_idx)) % seg_len
            # print(f"after modulo: {pos_idx=}")
            pos_idx[l:u] += pos_ub - seg_len
            # print(f"after subtract: {pos_idx=}")
            pos_ub = pos_ub - seg_len
            # print(f"pos idx for cascade: {i=} {pos_idx[l:u]=}")

        mask = None
        sink_mask = None
        if create_mask:
            mask = torch.zeros(self.max_seq_len,
                               device=self.device,
                               dtype=self.dtype)
            mask[end:] = torch.finfo(self.dtype).min
            mask = mask.view(self.max_batch_size, 1, 1, self.max_seq_len)
            pos_idx = self.pos_idx
            pos_idx[end:] = 0

            sink_mask = torch.zeros(self.num_sink_tokens,
                                    device=self.device,
                                    dtype=self.dtype)

            sink_mask = sink_mask.view(self.max_batch_size, 1, 1,
                                       self.num_sink_tokens)
        else:
            key_states = key_states[:, :, :end]
            value_states = value_states[:, :, :end]
            pos_idx = self.pos_idx[:end]

        return (
            self.get_cache("sink_keys", layer_idx, 0, self.num_sink_tokens),
            self.get_cache("sink_values", layer_idx, 0, self.num_sink_tokens),
            self.sink_pos.unsqueeze(0),
            sink_mask,
            key_states,
            value_states,
            pos_idx.unsqueeze(0) + self.num_sink_tokens,
            mask,
        )


def update_segment_pos(cascade_idx, pos, pos_ub, start_indices, stored_tokens,
                       l, u, seg_len, tmp_arange):
    u = torch.amin(
        torch.cat((u, l + torch.gather(stored_tokens, 0, cascade_idx))))

    seg_len = torch.amin(
        torch.cat((torch.gather(stored_tokens, 0,
                                cascade_idx).unsqueeze(0), seg_len)))
    start_idx = torch.gather(start_indices, 0, cascade_idx)

    tmp = (tmp_arange + (seg_len - start_idx)) % seg_len + (pos_ub - seg_len)
    pos.scatter_(1, l + tmp_arange.unsqueeze(0), tmp.unsqueeze(0))
    # pos[0, l:u] = (self.tmp_arange + (seg_len - start_idx)) % seg_len

    pos_ub.sub_(seg_len)
    return cascade_idx


def append_to_cache(cascade_idx, input_key_states, input_value_states,
                    input_score_states, keys, values, scores, mask, cache_idx,
                    score_idx, mask_idx, start_indices, stored_tokens, l, u,
                    seg_len, pos, pos_ub, tmp_arange):

    start_idx = torch.gather(start_indices, 0, cascade_idx)
    # we have empty room in this cache, so we need to shift the index
    # forward by the number of tokens already stored.
    stored = torch.gather(stored_tokens, 0, cascade_idx)
    s = start_idx + l + stored

    # we do not need to evict, find the end point and insert token
    # since this cache is not full, the insert point will be start + stored_tokens
    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s
    mask_idx_local = mask_idx * s

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)
    mask.scatter_(3, mask_idx_local, 0)

    stored_tokens.add_(F.one_hot(cascade_idx, stored_tokens.size(0)))

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, seg_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)

    return cascade_idx


def evict_from_cache(cascade_idx, input_key_states, input_value_states,
                     input_score_states, keys, values, scores, start_indices,
                     cache_idx, score_idx, l, u, segment_len, pos, pos_ub,
                     stored_tokens, tmp_arange):

    start_idx = torch.gather(start_indices, 0, cascade_idx)
    s = start_idx + l

    # we need to evict
    # 1. find the oldest token (start point), remove it and
    #    set input_key_state at that location

    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s

    next_input_key_state = torch.gather(keys, 2, cache_idx_local).clone()
    next_input_value_state = torch.gather(values, 2, cache_idx_local).clone()
    next_input_score_state = torch.gather(scores, 0, score_idx_local).clone()

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)

    # 2. rotate the start index.
    # new_start_idx = (start_idx + 1) % segment_len (vectorized version of this)
    # start_indices = (start_indices + F.one_hot(
    #     cascade_idx, start_indices.size(0))) % segment_len
    start_indices.add_(F.one_hot(cascade_idx, start_indices.size(0)))
    start_indices.fmod_(segment_len)

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, segment_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)

    # mask remains unchanged for this operation.
    return (cascade_idx, next_input_key_state, next_input_value_state,
            next_input_score_state)


def overwrite_cache(cascade_idx, input_key_states, input_value_states,
                    input_score_states, keys, values, scores, start_indices,
                    cache_idx, score_idx, l, u, seg_len, pos, pos_ub,
                    stored_tokens, tmp_arange):
    # print(
    #     f"{stored_tokens.size()=} {cascade_idx.size()=} {start_indices.size()=}"
    # )
    # print(f"{stored_tokens=} {cascade_idx=} {start_indices=}")
    start_idx = torch.gather(start_indices, 0, cascade_idx)
    # print("hit")
    stored = torch.gather(stored_tokens, 0, cascade_idx)
    # print("hit2")
    # print(f"{start_idx=} {stored=}")

    # s = start_idx + l
    # print(f"{start_idx - 1=} {(start_idx - 1) % stored=}")
    s = ((start_idx - 1) % stored) + l
    s = torch.amax(torch.cat((s, torch.zeros_like(s))))

    # print(f"{s=}")

    cache_idx_local = cache_idx * s
    score_idx_local = score_idx * s

    keys.scatter_(2, cache_idx_local, input_key_states)
    values.scatter_(2, cache_idx_local, input_value_states)
    scores.scatter_(0, score_idx_local, input_score_states)

    _ = update_segment_pos(cascade_idx, pos, pos_ub, start_indices,
                           stored_tokens, l, u, seg_len, tmp_arange)

    # move along cascade idx for the next iteration
    cascade_idx.add_(1)
    return cascade_idx


def add_sinks(input_key_states, input_value_states, sink_keys, sink_values,
              sink_pos, sink_mask, stored_sinks, cache_idx, pos_idx, mask_idx):

    cache_idx_local = cache_idx * stored_sinks
    pos_idx_local = pos_idx * stored_sinks
    mask_idx_local = mask_idx * stored_sinks

    sink_keys.scatter_(2, cache_idx_local, input_key_states)
    sink_values.scatter_(2, cache_idx_local, input_value_states)
    sink_pos.scatter_(1, pos_idx_local, stored_sinks.expand_as(pos_idx_local))
    sink_mask.scatter_(3, mask_idx_local, 0)

    return input_key_states


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
        self.do_cache_cpu = torch.tensor([True for _ in range(self.cascades)],
                                         dtype=torch.bool,
                                         requires_grad=False)
        self.do_cache = torch.tensor([True for _ in range(self.cascades)],
                                     device=device,
                                     dtype=torch.bool,
                                     requires_grad=False)

        print(f"{self.cascades=} {self.do_cache=}")
        self.do_cache_every_n = torch.tensor(
            [2**i for i in range(self.cascades)],
            dtype=torch.long,
            requires_grad=False,
        )

        self.beta = np.exp(-np.log(100) / window_length)
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = torch.tensor(
            0, dtype=torch.long, requires_grad=False
        )  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.stored_sinks = 0

        self.stored_tokens = torch.tensor([0 for _ in range(self.cascades)],
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

        self.pos_ub = torch.tensor(max_seq_len,
                                   device=device,
                                   dtype=torch.long,
                                   requires_grad=False)

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

        self.scalar = torch.ones(1,
                                 device=self.device,
                                 dtype=torch.long,
                                 requires_grad=False)

        self.cascade_idx = torch.tensor(0,
                                        device=self.device,
                                        dtype=torch.long,
                                        requires_grad=False)

        self.cascade_bounds = []
        for i in range(self.cascades):
            self.cascade_bounds.append(
                (self.scalar * self.window_length * i,
                 self.scalar * self.window_length * (i + 1),
                 self.scalar * self.window_length, self.window_length))

        self.cache_idx = torch.ones(self.max_batch_size,
                                    self.heads,
                                    1,
                                    self.dim,
                                    device=self.device,
                                    dtype=torch.long,
                                    requires_grad=False)

        self.pos_idx = torch.ones(1,
                                  1,
                                  device=self.device,
                                  dtype=torch.long,
                                  requires_grad=False)

        self.mask_idx = torch.ones(1,
                                   1,
                                   1,
                                   1,
                                   device=self.device,
                                   dtype=torch.long,
                                   requires_grad=False)

        self.score_idx = torch.ones(1,
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

        self.score_states = torch.zeros(1,
                                        device=self.device,
                                        dtype=self.dtype,
                                        requires_grad=False)

    def set_cache_bools(self):
        # minus one because seen tokens is incremented before tokens are really added. Therefore we need to subtract that one
        for i, _ in enumerate(self.do_cache_cpu):
            if (self._seen_tokens - 1 -
                    self.num_sink_tokens) % self.do_cache_every_n[i] == 0:
                self.do_cache_cpu[i] = True
                continue

            self.do_cache_cpu[i] = False
        self.do_cache.copy_(self.do_cache_cpu)

    def get_cascade_bounds(self, i):
        return self.cascade_bounds[i]

    def get_seq_length(self,
                       layer_idx: Optional[int] = 0,
                       cascade_idx: Optional[int] = -1) -> int:
        return sum([v for v in self.stored_tokens])

    def get_max_length(self) -> Optional[int]:
        return self.max_seq_len

    def update_attention_scores(self, scores, layer_idx) -> None:
        self.score_cache = self.beta * self.score_cache + (1 -
                                                           self.beta) * scores

    def warn(self, args):
        warnings.warn(
            "the cascading cache is full, evicted context from the last cascade will be dropped"
        )
        return args

    def add_keys(self, input_key_states, input_value_states):

        # in order to create the positional embeddings in teh same loop as
        # the main logic, we must know if we are going to add anything to the
        # cache or not which will change what happens to the positional embeddings.
        stored_tokens_cpu = self.stored_tokens.cpu()

        tmp_pos_ub = stored_tokens_cpu.sum()
        add_to_cache = self.do_cache_cpu.sum(
        ) * self.window_length > tmp_pos_ub
        eager_add = self.do_cache_cpu.sum() * self.window_length == tmp_pos_ub
        if add_to_cache or eager_add:
            tmp_pos_ub += 1

        self.pos_ub.fill_(tmp_pos_ub.item())

        input_score_states = self.score_states

        self.cascade_idx.zero_()
        for i in range(self.cascades):
            l, u, segment_len, segment_len_cpu = self.get_cascade_bounds(i)

            if self.do_cache_cpu[i]:
                if stored_tokens_cpu[i] < segment_len_cpu:

                    _ = self.append_to_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.mask, self.cache_idx,
                        self.score_idx, self.mask_idx, self.start_indices,
                        self.stored_tokens, l, u, segment_len, self.pos,
                        self.pos_ub, self.tmp_arange)
                    break
                else:
                    (_, input_key_states, input_value_states,
                     input_score_states) = self.evict_from_cache(
                         self.cascade_idx, input_key_states,
                         input_value_states, input_score_states,
                         self.key_cache, self.value_cache, self.score_cache,
                         self.start_indices, self.cache_idx, self.score_idx, l,
                         u, segment_len, self.pos, self.pos_ub,
                         self.stored_tokens, self.tmp_arange)

                    if i + 1 > (self.cascades - 1):
                        break
            else:
                if stored_tokens_cpu[i] == 0:
                    # if we are not supposed to move the cache, but we were called
                    # with states as an input. Then there are two possibilities:
                    # 1. We are not supposed to do cache, but the length of this cache is zero.
                    #    this may happen due to the do_cache input_values not lining up perfectly with powers of 2.
                    #    In this case, we should add an element to the cache so it doesn't just get automatically evicted.
                    _ = self.append_to_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.mask, self.cache_idx,
                        self.score_idx, self.mask_idx, self.start_indices,
                        self.stored_tokens, l, u, segment_len, self.pos,
                        self.pos_ub, self.tmp_arange)
                    break
                else:
                    # 2. Since we know this cache has something in it, and we are not to do caching,
                    #    find the oldest thing in this cache, compare attention input_scores,
                    #    and remove if needed.

                    s = self.start_indices[i].add(l)

                    score_idx = self.score_idx * s
                    old_input_score = torch.gather(self.score_cache, 0,
                                                   score_idx)
                    if old_input_score > input_score_states:
                        # old input_score is better, do nothing.
                        # increment cascade index for next iter
                        # break onstead of cotinue because this stops the cascade
                        break

                    _ = self.overwrite_cache(
                        self.cascade_idx, input_key_states, input_value_states,
                        input_score_states, self.key_cache, self.value_cache,
                        self.score_cache, self.start_indices, self.cache_idx,
                        self.score_idx, l, u, segment_len, self.pos,
                        self.pos_ub, self.stored_tokens, self.tmp_arange)

                    break

        return

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

        if self.stored_sinks < self.num_sink_tokens:
            _ = self.add_sinks(key_states, value_states, self.sink_keys,
                               self.sink_values, self.sink_pos, self.sink_mask,
                               self.stored_sinks * self.scalar, self.cache_idx,
                               self.pos_idx, self.mask_idx)

            self.stored_sinks += 1
            pos = self.pos
        else:
            self.add_keys(key_states, value_states)

            # self.pos[:, self.stored_tokens.sum():] = -self.num_sink_tokens
            pos = self.pos.add(self.num_sink_tokens)
        # print(f"\n\n\nbefore")
        # print(
        #     f"{self.sink_keys=}\n{self.sink_values=}\n{self.sink_pos=}\n{self.sink_mask=}"
        # )

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
            pos,
            self.mask,
        )


class CascadingSinkCachOriginal(SinkCache):
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

    def __init__(self,
                 window_length: int,
                 num_sink_tokens: int,
                 cascades: int = 8) -> None:
        self.key_cache: List[List[torch.Tensor]] = [[]
                                                    for _ in range(cascades)]
        self.attn_score_cache: List[List[torch.Tensor]] = [
            [] for _ in range(cascades)
        ]
        self.value_cache: List[List[torch.Tensor]] = [[]
                                                      for _ in range(cascades)]

        self.cascades = cascades
        # self.cache_lens = [window_length // 2**i for i in range(cascades)]
        self.cache_lens = [
            window_length if i < cascades - 1 else window_length -
            num_sink_tokens for i in range(cascades)
        ]
        print(f"{self.cache_lens=}")

        self.sink_keys: List[torch.Tensor] = []
        self.sink_values: List[torch.Tensor] = []

        self.do_cache = [True for _ in range(cascades)]
        self.do_cache_every_n = [2**i for i in range(cascades)]
        self.beta = 0.99
        # self.do_cache_every_n = [1, 2**7]
        # self.do_cache_every_n = [1, 2**4, 2**8, 2**12]
        self.num_sink_tokens = num_sink_tokens

        self.window_length = window_length
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def set_cache_bools(self):
        for i, _ in enumerate(self.do_cache):
            if self._seen_tokens % self.do_cache_every_n[i] == 0:
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
                for i, _ in enumerate(self.cache_lens)
            ])

        if len(self.key_cache[cascade_idx]) <= layer_idx:
            return 0
        return self.key_cache[cascade_idx][layer_idx].shape[-2]

    def get_max_length(self, cascade_idx: int = 0) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.cache_lens[cascade_idx]

    def update_attention_scores(
        self,
        attention_scores: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        if self.seen_tokens <= self.num_sink_tokens:
            return

        start, end = self.num_sink_tokens, self.num_sink_tokens
        for i, _ in enumerate(self.key_cache):
            start = end
            end = end + self.cache_lens[i]

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

            if len(self.key_cache[i]) <= layer_idx:
                # Empty cache
                if self.do_cache[i] or self.get_seq_length() < sum(
                        self.cache_lens):
                    self.key_cache[i].append(key_states)
                    self.value_cache[i].append(value_states)
                    self.attn_score_cache[i].append(score_states.clone())

                break

            elif key_states.shape[-2] + self.get_seq_length(
                    layer_idx, i) <= self.get_max_length(i):
                # Growing cache
                if self.do_cache[i] or self.get_seq_length() < sum(
                        self.cache_lens):
                    self.key_cache[i][layer_idx] = torch.cat(
                        [self.key_cache[i][layer_idx], key_states], dim=-2)

                    self.value_cache[i][layer_idx] = torch.cat(
                        [self.value_cache[i][layer_idx], value_states], dim=-2)

                    self.attn_score_cache[i][layer_idx] = torch.cat(
                        (self.attn_score_cache[i][layer_idx],
                         score_states.clone()),
                        dim=-1)
                break

            else:
                if not self.do_cache[i] and self.get_seq_length() >= sum(
                        self.cache_lens):
                    # we have the evcited tokens from the last cascade layer,
                    # so compare them to the last tokens in this layer and keep the
                    # ones with a larger total attention score.
                    prev_score = self.attn_score_cache[i][layer_idx][:, -1:]
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
                keys_to_keep = key_cache[:, :, -self.get_max_length(i) +
                                         key_states.shape[-2]:]

                scores_to_keep = score_cache[:, -self.get_max_length(i) +
                                             key_states.shape[-2]:]

                values_to_keep = value_cache[:, :, -self.get_max_length(i) +
                                             value_states.shape[-2]:]

                # print(f"catting cache: {i}")
                self.key_cache[i][layer_idx] = torch.cat(
                    [keys_to_keep, key_states], dim=-2)
                self.value_cache[i][layer_idx] = torch.cat(
                    [values_to_keep, value_states], dim=-2)
                self.attn_score_cache[i][layer_idx] = torch.cat(
                    (scores_to_keep, score_states.clone()), dim=-1)

                # these are the evicted tokens for the next iteration
                key_states = key_cache[:, :, :-self.get_max_length(i) +
                                       key_states.shape[-2]]
                score_states = score_cache[:, :-self.get_max_length(i) +
                                           key_states.shape[-2]]
                value_states = value_cache[:, :, :-self.get_max_length(i) +
                                           value_states.shape[-2]]

        out_keys, out_values = [], []
        for i in reversed(range(len(self.key_cache))):
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
                    if prev_score[0, 0] / (1 - self.beta) > score_states[
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
                    if prev_score[0, 0] / (1 - self.beta) > score_states[
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


def test_non_compiled():
    window, sink = 2, 2
    dim, head, layers = 1, 1, 1
    device = "cuda:7"
    dtype = torch.float16
    cache = CascadingSinkCache(
        window_length=window,
        num_sink_tokens=sink,
        max_batch_size=1,
        heads=head,
        dim=dim,
        n_layers=layers,
        device="cuda:7",
        dtype=dtype,
        max_seq_len=100,
    )

    slow_cache = CascadingSinkCacheSlow(window_length=window,
                                        num_sink_tokens=sink)

    slow_times, fast_times = [], []
    for i in range(1000):
        for layer_idx in range(layers):
            if i < sink:
                k, v = torch.ones(
                    1, head, 1, dim, device=device,
                    dtype=dtype) * (i + 1), torch.ones(
                        1, head, 1, dim, device=device, dtype=dtype) * (i + 1)
            else:
                k, v = torch.ones(
                    1, head, 1, dim, device=device,
                    dtype=dtype) * (i + 1), torch.ones(
                        1, head, 1, dim, device=device, dtype=dtype) * (i + 1)

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

            pos = pos.squeeze(0)
            if k_nosink is not None:
                # print(f"{k.size()=} {k_nosink.size()=}")
                k, v = torch.cat((k, k_nosink), dim=-2), torch.cat(
                    (v, v_nosink), dim=-2)
                pos = torch.cat((pos, pos_nosink.squeeze(0)), dim=-1)

            # print(
            #     f"before sort: \n{slow_k.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}"
            # )
            print(f"{pos=}")

            argsort = torch.argsort(pos)
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
                    f"{slow_k.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - slow_k).abs().view(-1)}"
                )
                exit("too big")

            # print(
            #     f"output for {layer_idx=} {sk.size()=} {sv.size()=} {spos.size()=}"
            # )
    slow_times = sum(slow_times) / len(slow_times)
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{slow_times=} {fast_times=}")


def compile_cache(
    cache,
    window,
    sink,
    dim,
    head,
    max_seq,
    device,
    dtype,
):

    print(f"{window=} {sink=} {dim=} {head=} {max_seq=} {device=} {dtype=}")
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

    cascade_idx = cache.cascade_idx
    sink_keys = cache.sink_keys
    sink_values = cache.sink_values
    sink_pos = cache.sink_pos
    sink_mask = cache.sink_mask

    keys = cache.key_cache
    values = cache.value_cache
    scores = cache.score_cache
    pos = cache.pos
    pos_ub = cache.pos_ub

    mask = cache.mask
    stored_sinks = cache.stored_sinks * cache.scalar

    start_indices = cache.start_indices
    stored_tokens = cache.stored_tokens
    cache_idx = cache.cache_idx
    mask_idx = cache.mask_idx
    pos_idx = cache.pos_idx
    score_idx = cache.score_idx

    l = cache.scalar * 0
    u = cache.scalar * window
    segment_len = cache.scalar * window
    tmp_arange = cache.tmp_arange

    # append_to_cache(cascade_idx, input_key_states, input_value_states,
    #                 input_score_states, keys, values, scores, mask, cache_idx,
    #                 score_idx, mask_idx, start_indices, stored_tokens, l, u,
    #                 segment_len, pos, pos_ub, tmp_arange)

    # evict_from_cache(cascade_idx, input_key_states, input_value_states,
    #                  input_score_states, keys, values, scores, start_indices,
    #                  cache_idx, score_idx, l, u, segment_len, pos, pos_ub,
    #                  stored_tokens, tmp_arange)

    # overwrite_cache(cascade_idx, input_key_states, input_value_states,
    #                 input_score_states, keys, values, scores, start_indices,
    #                 cache_idx, score_idx, l, u, segment_len, pos, pos_ub,
    #                 stored_tokens, tmp_arange)

    # add_sinks(input_key_states, input_value_states, sink_keys, sink_values,
    #           sink_pos, sink_mask, stored_sinks, cache_idx, pos_idx, mask_idx)

    # print(
    #     f"before segment pos {cache.stored_tokens=} {cache.pos=} {cache.tmp_arange=} {segment_len=} {l=} {u=}"
    # )
    # seg_len = stored_tokens[cascade_idx]
    # pos_ub = pos_ub.fill_(stored_tokens.sum())
    # print(
    #     f"{cache.tmp_arange=} {seg_len=} {start_indices[cascade_idx]=} {pos_ub=}"
    # )
    # print(
    #     f"{(cache.tmp_arange + (seg_len - start_indices[cascade_idx])) % seg_len + (pos_ub - seg_len)}"
    # )

    # update_segment_pos(cascade_idx, pos, pos_ub, start_indices, stored_tokens,
    #                    l, u, segment_len, tmp_arange)

    # print(f"after {cache.stored_tokens=} {cache.pos=}")
    # cache.stored_tokens.zero_()
    # cache.pos.zero_()
    # print(f"after reset {cache.stored_tokens=} {cache.pos=}")
    # exit()

    cache.append_to_cache = make_graphed_callables(
        append_to_cache,
        (cascade_idx, input_key_states, input_value_states, input_score_states,
         keys, values, scores, mask, cache_idx, score_idx, mask_idx,
         start_indices, stored_tokens, l, u, segment_len, pos, pos_ub,
         tmp_arange),
        allow_unused_input=True)

    cascade_idx.zero_()
    cache.evict_from_cache = make_graphed_callables(
        evict_from_cache,
        (cascade_idx, input_key_states, input_value_states, input_score_states,
         keys, values, scores, start_indices, cache_idx, score_idx, l, u,
         segment_len, pos, pos_ub, stored_tokens, tmp_arange),
        allow_unused_input=True)

    cascade_idx.zero_()
    cache.overwrite_cache = make_graphed_callables(
        overwrite_cache,
        (cascade_idx, input_key_states, input_value_states, input_score_states,
         keys, values, scores, start_indices, cache_idx, score_idx, l, u,
         segment_len, pos, pos_ub, stored_tokens, tmp_arange),
        allow_unused_input=True)

    cascade_idx.zero_()
    cache.add_sinks = make_graphed_callables(
        add_sinks,
        (input_key_states, input_value_states, sink_keys, sink_values,
         sink_pos, sink_mask, stored_sinks, cache_idx, pos_idx, mask_idx),
        allow_unused_input=True)

    # cache.update_segment_pos = make_graphed_callables(
    #     update_segment_pos, (cascade_idx, pos, pos_ub, start_indices,
    #                          stored_tokens, l, u, segment_len, tmp_arange),
    #     allow_unused_input=True)

    cache.stored_tokens.zero_()
    cache.pos.zero_()
    cache.mask.fill_(torch.finfo(dtype).min)
    cache.key_cache.zero_()
    cache.value_cache.zero_()
    cache.score_cache.zero_()
    cache.start_indices.zero_()
    cache.sink_keys.zero_()
    cache.sink_values.zero_()
    cache.sink_pos.zero_()
    cache.sink_mask.fill_(torch.finfo(dtype).min)

    return cache


def test_nsys():
    window, sink = 512, 4
    dim, head, layers = 2048 // 16, 16, 1
    max_seq = 2048
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

    cache = compile_cache(
        cache,
        window,
        sink,
        dim,
        head,
        max_seq,
        device,
        dtype,
    )

    with torch.no_grad():
        for i in range(100):
            for layer_idx in range(layers):
                # print(f"{'='*50}")
                k, v = torch.ones(
                    1, head, 1, dim, device=device,
                    dtype=dtype) * (i + 1), torch.ones(
                        1, head, 1, dim, device=device, dtype=dtype) * (i + 1)

                # print(f"\n\n\n\ninput for {layer_idx=} kv {k.size()=} {v.size()=}")

                k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask = cache.update(
                    k, v, layer_idx=layer_idx)


def test_compiled_non_compiled():
    window, sink = 2, 2
    dim, head, layers = 1, 1, 1
    max_seq = window * 20
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

    cache = compile_cache(
        cache,
        window,
        sink,
        dim,
        head,
        max_seq,
        device,
        dtype,
    )

    # cache = torch.compile(cache, mode="reduce-overhead", fullgraph=True)

    slow_cache = CascadingSinkCache(
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

    slow_dumb_cache = CascadingSinkCacheSlow(
        window_length=window,
        num_sink_tokens=sink,
    )

    with torch.no_grad():
        slow_times, fast_times, slow_dumb_times = [], [], []
        for i in range(6000):
            for layer_idx in range(layers):
                # print(f"{'='*50}")
                k, v = torch.ones(
                    1, head, 1, dim, device=device,
                    dtype=dtype) * (i + 1), torch.ones(
                        1, head, 1, dim, device=device, dtype=dtype) * (i + 1)

                tic = time.perf_counter()
                k_nocomp, v_nocomp = slow_dumb_cache.update(
                    k.clone(), v.clone(), layer_idx=layer_idx)
                slow_dumb_times.append(time.perf_counter() - tic)

                tic = time.perf_counter()
                k_nocomp, v_nocomp, pos_nocomp, sink_mask_nocomp, k_nosink_nocomp, v_nosink_nocomp, pos_nosink_nocomp, mask_nocomp = slow_cache.update(
                    k, v, layer_idx=layer_idx)
                slow_times.append(time.perf_counter() - tic)
                n = pos_nocomp.squeeze(0).size(0)

                if k_nosink_nocomp is not None:
                    # print(f"\n\n{k.size()=} {k_nosink.size()=}")
                    k_nocomp, v_nocomp = torch.cat(
                        (k_nocomp, k_nosink_nocomp), dim=-2), torch.cat(
                            (v_nocomp, v_nosink_nocomp), dim=-2)
                    pos_nocomp = torch.cat((pos_nocomp, pos_nosink_nocomp),
                                           dim=-1).squeeze(0)
                    n = pos_nocomp.size(0)
                    k_nocomp, v_nocomp = k_nocomp[:, :, :n], v_nocomp[:, :, :n]

                argsort = torch.argsort(pos_nocomp.squeeze(0)[:n])

                # print(f"{k_nocomp[0, 0, :, 0]=}")
                k_nocomp, v_nocomp = k_nocomp[:, :, argsort], v_nocomp[:, :,
                                                                       argsort]

                # ============================================================================================
                tic = time.perf_counter()
                k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask = cache.update(
                    k, v, layer_idx=layer_idx)
                fast_times.append(time.perf_counter() - tic)

                # print(f"\n\n{k.size()=} {k_nosink.size()=}")
                k, v = torch.cat((k, k_nosink), dim=-2), torch.cat(
                    (v, v_nosink), dim=-2)
                pos = torch.cat((pos, pos_nosink), dim=-1).squeeze(0)

                # print(f"{k[0, 0, :,  0]=}")
                mask = torch.cat((sink_mask, mask), dim=-1)

                n = (mask == 0).sum()
                k, v = k[:, :, :n], v[:, :, :n]
                argsort = torch.argsort(pos[:n])
                # print(f"{mask=} {k=} {v=}")

                # print(
                #     f"before sort: \n{slow_k.view(-1)=}\n{k.reshape(-1)=}\n{pos.reshape(-1)=}"
                # )

                k, v = k[:, :, argsort], v[:, :, argsort]

                # print(f"after sort: {k.view(-1)}")
                if not k_nocomp.size() == k.size():
                    print(f"{k_nocomp.size()=} {k.size()=}")
                    print(f"sizes not equal...\n{k_nocomp=} {k=}")
                    exit()

                diff = (k_nocomp - k).abs().amax()
                print(f"k diff: {diff=} {i=} {layer_idx=}")
                if diff > 1e-6:
                    print(
                        f"{k_nocomp.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_nocomp).abs().view(-1)=}"
                    )
                    exit("too big")

        # print(
        #     f"output for {layer_idx=} {sk.size()=} {sv.size()=} {spos.size()=}"
        # )
    slow_dumb_times = sum(slow_dumb_times) / len(slow_dumb_times)
    slow_times = sum(slow_times) / len(slow_times)
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{slow_dumb_times=} {slow_times=} {fast_times=}")


def test_compiled():
    window, sink = 10, 4
    dim, head, layers = 128, 32, 1
    max_seq = 250
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

    cache = compile_cache(cache)
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
                print(f"comp {pos=}")

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

            print(
                f"output for {layer_idx=} {sk.size()=} {sv.size()=} {spos.size()=}"
            )
    slow_times = sum(slow_times) / len(slow_times)
    fast_times = sum(fast_times) / len(fast_times)
    print(f"{slow_times=} {fast_times=}")


if __name__ == "__main__":
    # test_gs_slice()
    # test_non_compiled()
    # test_compiled()
    test_compiled_non_compiled()
    # test_nsys()
