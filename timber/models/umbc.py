import math

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from typing import Any, List, Optional, Tuple

T = torch.Tensor
OT = Optional[T]
LT = List[T]
OLT = Optional[LT]

EPS = 1e-20


class GradCorrecter(nn.Module):

    def register_grad_correct_hooks(self, grad_size: int,
                                    set_size: int) -> Any:

        def backward_hook(g: T) -> T:
            return (set_size / grad_size) * g

        handles = []
        for n, p in self.named_parameters():
            if "norm_after" not in n:
                h = p.register_hook(backward_hook)
                handles.append(h)

        def remove() -> None:
            [h.remove() for h in handles]

        return remove


class MBCFunction(GradCorrecter):
    norm_after: nn.Module

    def partitioned_forward(self, X: T, X_nograd: T) -> T:
        """for training the model where x_nograd has no gradient."""
        raise NotImplementedError()

    def pre_forward_mbc(self) -> None:
        """
        for pre processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward_mbc(self,
                    X: T,
                    X_prev: OT = None,
                    c_prev: OT = None,
                    grad: bool = True,
                    mask: OT = None) -> Tuple[T, OT]:
        """
        for training the model with minibatches.
        X_prev: is the previous state at timestep (t-1)
        grad: whether the gradient is computed on this chunk.
        Returns: The current state at timestep (t)
        """
        raise NotImplementedError()

    def post_forward_mbc(self, X: T, c: OT = None, mask: OT = None) -> T:
        """
        for post processing the minibatched forward pooled
        vector and normalization constant
        """
        raise NotImplementedError()

    def forward(self, X: T) -> T:
        raise NotImplementedError()

    def grad_correct(self, c: float) -> None:
        raise NotImplementedError()


def set_preact_value(W: T, mask: OT = None, val: float = -1e20) -> T:
    if mask is None:
        return W

    # mask is (batch * heads, 1, set size). The middle dimension
    # get projected to all slots in the that dimension
    return W.masked_fill(mask == 0, val)


def apply_slot_normalization_with_mask(W: T,
                                       mask: OT = None,
                                       eps: float = EPS) -> T:
    if mask is None:
        return W / (W.sum(dim=-2, keepdim=True) + eps)

    W = W.masked_fill(mask == 0, 0.0)
    return W / (W.sum(dim=-2, keepdim=True) + eps)


def update_set_normalization_constant(W: T,
                                      mask: OT = None,
                                      eps: float = EPS) -> T:
    if mask is None:
        return W.sum(dim=-1, keepdim=True)

    # mask is (batch * heads, 1, set size).
    W = W.masked_fill(mask == 0, 0.0)
    return W.sum(dim=-1, keepdim=True)


def softmax_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    # normalization over the set dimension happens within the SSE
    W = set_preact_value(W, mask=mask).clamp(max=10)
    return torch.exp(W)


def sigmoid_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    return torch.sigmoid(W)


def slot_sigmoid_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = torch.sigmoid(W)
    return apply_slot_normalization_with_mask(W, mask, eps=eps)


def slot_softmax_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = W.softmax(dim=-2)
    # its possible that every value in the set was set to the
    # null value so we have to reset them all to zero to be safe
    if mask is not None:
        W = W.masked_fill(mask == 0, 0.0)
    return W


def slot_exp_attn_act(W: T, eps: float = EPS, mask: OT = None) -> T:
    W = set_preact_value(W, mask=mask)
    W = torch.exp(W - W.amax(dim=-2, keepdim=True))
    # its possible that every value in the set was set to the
    # null value so we have to reset them all to zero to be safe
    if mask is not None:
        W = W.masked_fill(mask == 0, 0.0)
    return W


attn_act_funcs = {
    "softmax": softmax_attn_act,
    "sigmoid": sigmoid_attn_act,
    "slot-sigmoid": slot_sigmoid_attn_act,
    "slot-softmax": slot_softmax_attn_act,
    "slot-exp": slot_exp_attn_act
}


class Slots(nn.Module):

    def __init__(self,
                 K: int,
                 h: int,
                 slot_type: str,
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.name = "Slots"
        self.K = K  # Number of Slots
        self.h = h  # Slot size
        self.slot_type = slot_type  # Deterministic or Random

        if slot_type not in ["random", "deterministic"]:
            raise ValueError("{} not implemented for slots".format(
                self.slot_type))

        if slot_type == "random":
            # same initialization as "Weight Uncertainty in Neural Networks"
            self.mu = nn.Embedding(self.K, self.h, **factory_kwargs)
            self.sigma = nn.Embedding(self.K, self.h, **factory_kwargs)

            return

        self.weight = nn.Embedding(self.K, self.h, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        if self.slot_type == "random":
            nn.init.uniform_(self.mu.weight, -0.2, 0.2)
            nn.init.uniform_(self.sigma.weight, -5.0, -4.0)
            return

        nn.init.xavier_uniform_(self.weight.weight)  # type: ignore

    def forward(self):
        return self.sample_s()

    def sample_s(self) -> T:
        if self.slot_type == "random":
            if self.training:
                return (  # type: ignore
                    torch.randn_like(self.mu(0)) * \
                    F.softplus(self.sigma(0)) + self.mu
                ).unsqueeze(0)
            return self.mu.unsqueeze(0)

        # returning the parameter S caused problems because it is an
        # nn.Parameter and the above random returns a tensor. Return
        # a tensor here to make it consistent
        return self.weight.weight.unsqueeze(0)


class SlotSetEncoder(MBCFunction):

    def __init__(self,
                 K: int,
                 dim: int,
                 slot_type: str = "random",
                 ln_slots: bool = True,
                 heads: int = 4,
                 bias: bool = True,
                 slot_drop: float = 0.0,
                 attn_act: str = "slot-sigmoid",
                 eps: float = EPS,
                 ln_after: bool = True,
                 max_batch: int = 32):
        super().__init__()
        self.name = "SlotSetEncoder"
        self.dim = dim
        self.eps = eps  # Additive epsilon for stability
        self.heads = heads  # number of attention heads
        self.bias = bias
        self.ln_after = ln_after
        self.ln_slots = ln_slots  # never used
        self.slot_drop = slot_drop
        self.attn_act = attn_act
        self.K = K
        self.max_batch = max_batch

        self.slots: Slots

        if dim % heads != 0:
            raise ValueError(
                f"for multihead attention, {dim} must be evenly divisible by {heads}"
            )  # noqa

        # Normalization Term
        self.sqrt_dim = 1.0 / math.sqrt(dim // heads)

        self.slots = nn.Embedding(K, dim)
        # self.slots = Slots(K=K, h=dim, slot_type=slot_type)

        # self.sse_q = nn.Linear(dim, dim, bias=bias)
        self.sse_v = nn.Linear(dim, dim, bias=bias)
        self.sse_k = nn.Linear(dim, dim, bias=bias)

        self.norm_slots = nn.LayerNorm(
            normalized_shape=dim) if ln_slots else nn.Identity()

        self.norm_after = nn.LayerNorm(dim) if ln_after else nn.Identity()
        self.init_cache()

    def init_cache(self):
        self.register_buffer(
            "x_prev",
            torch.zeros(self.max_batch,
                        self.heads,
                        self.K,
                        self.dim // self.heads,
                        device=self.sse_v.weight.device))
        self.register_buffer(
            "c_prev",
            torch.zeros(self.max_batch,
                        self.heads,
                        self.K,
                        1,
                        device=self.sse_v.weight.device))

    def sample_s(self) -> T:
        S = self.slots.weight.unsqueeze(0)
        S = self.norm_slots(S)
        return S

    def head_split(self, x):
        # in: (B, S, D) --> (B, H, S, D//H)
        return x.reshape(x.size(0), x.size(1), self.heads,
                         x.size(2) // self.heads).transpose(1, 2)

    def get_attn_v(self, X: T, S: OT = None) -> Tuple[T, T, T]:
        if S is None:  # S \in R^{B x K xh}
            S = self.sample_s().repeat(X.size(0), 1, 1)
        assert S is not None

        if S.size(0) == 1:  # in case S was passed in and not repeated
            S = S.repeat(X.size(0), 1, 1)

        Q, K, V = S, self.sse_k(X), self.sse_v(X)  # (B, T, D)
        Q, K, V = map(self.head_split, (Q, K, V))  # (B, H, T, D)

        A = torch.einsum("bhkd,bhsd->bhks", Q * np.sqrt(self.sqrt_dim),
                         K * np.sqrt(self.sqrt_dim))

        return Q, A, V

    def forward(self, X: T, S: OT = None, mask: OT = None) -> T:
        B, T, D = X.size()

        if mask is not None:
            mask = mask.repeat(self.heads, 1).unsqueeze(1)

        S, W, V = self.get_attn_v(X, S=S)  # W: (B, H, K, S) V: (B, H, S, D)

        W = W.softmax(dim=-2)  # softmax over the slots
        C = W.sum(dim=-1, keepdim=True)  # cumulative normalization constant

        A = W / (C + self.eps)

        S_hat = torch.einsum("bhks,bhsd->bhkd", A, V)  # (B, H, K, D // H)

        S_hat = S_hat.permute(0, 2, 1, 3)  # (B, S, K, H, D//H)
        B, K, H, D = S_hat.size()
        S_hat = S_hat.reshape(B, K, H * D)

        S_hat = self.norm_after(S_hat)
        return S_hat  # type: ignore

    def forward_cumulative(self,
                           X: T,
                           S: OT = None,
                           mask: OT = None,
                           chunk_size=1) -> T:
        """
        X: (B, S, D)
        output: (B, S, K, D)

        items are cumulatively pooled so that the last output element contains
        information from every element in the sequence.
        """
        B, T, D = X.size()
        if T % chunk_size != 0:
            raise ValueError(f"{T % chunk_size=} must be zero")
        T, c = T // chunk_size, chunk_size

        X = X.reshape(B * T, c, D)

        if mask is not None:
            raise NotImplementedError(
                "masking has not been implemented for umbc cumulative")

        S, W, V = self.get_attn_v(X, S=S)
        # W: (B * S, H, K, 1), C: (B * S, H, K, 1), V: (B * S, H, 1, D//H)

        W = W.reshape(B, T, self.heads, self.K, c)
        V = V.reshape(B, T, self.heads, c, D // self.heads)

        W = W.softmax(dim=-2)  # softmax over the slots
        C = W.sum(dim=-1, keepdim=True).cumsum(
            dim=1)  # cumulative normalization constant

        A = torch.einsum("bshkc,bshcd->bshkd", W, V)
        A = A.cumsum(dim=1)
        S_hat = A / (C + self.eps)

        S_hat = S_hat.transpose(2,
                                3)  # (B, T, H, K, D//H) --> (B, T, K, H, D//H)
        S_hat = S_hat.reshape(B, T, self.K, D)

        S_hat = self.norm_after(S_hat)
        return S_hat  # type: ignore

    def forward_mbc(
        self,
        X: T,
        grad: bool = True,
        mask: OT = None,
    ) -> Tuple[T, T]:
        if mask is not None:
            mask = mask.repeat(self.heads, 1).unsqueeze(1)

        s = self.sample_s().repeat(X.size(0), 1, 1)
        with torch.set_grad_enabled(grad):
            _, x, c = self.process_batch(X, S=s, mask=mask)

        with torch.no_grad():
            c_prev, x_prev = torch.clone(self.c_prev[:c.size(0)]), torch.clone(
                self.x_prev[:x.size(0)])
            self.x_prev[:x.size(0)].add_(x)
            self.c_prev[:c.size(0)].add_(c)

        c += c_prev
        x += x_prev

        view_std = (x.size(0), s.size(1), -1)

        x = x.transpose(1, 2) / (c.transpose(1, 2) + self.eps)  # type: ignore
        x = x.reshape(*view_std)

        x = self.norm_after(x)
        return x

    def process_batch(self,
                      X: T,
                      S: OT = None,
                      mask: OT = None) -> Tuple[T, T, T]:
        """
        this is a 'partial forward' which allows the user to aggreate the
        components manually returns:
        S: Slots
        S_hat: SSE output to be aggregated
        C: normalization constant
        """
        S, W, V = self.get_attn_v(X, S=S)

        W = W.softmax(dim=-2)  # softmax over the slots
        C = W.sum(dim=-1, keepdim=True)  # cumulative normalization constant

        S_hat = torch.einsum("bhks,bhsd->bhkd", W, V)  # (B, H, K, D // H)

        return S, S_hat, C  # type: ignore

    def post_forward_mbc_cleanup(self):
        with torch.no_grad():
            self.x_prev.zero_()
            self.c_prev.zero_()

    def grad_correct(self, c: float) -> None:
        for n, p in self.named_parameters():
            if "norm_after" not in n:
                p.grad.data.mul_(c)


if __name__ == "__main__":
    sse = SlotSetEncoder(
        K=16,
        dim=128,
        slot_type="deterministic",
        bias=False,
    )
    sse.eval()

    x = torch.randn(32, 256, 128)
    out = sse(x)
    out_cumulative = sse.forward_cumulative(x)
    last = out_cumulative[:, -1]

    diff = (out - last).abs().amax()
    print(f"{out.size()=} {last.size()=}")
    print(f"max diff: {diff=}")

    sse.init_cache()
    out_mbc = sse.forward_mbc(x)
    sse.post_forward_mbc_cleanup()

    diff = (out - out_mbc).abs().amax()
    print(f"{out.size()=} {out_mbc.size()=}")
    print(f"max diff: {diff=}")

    chnked_out = []
    for chnk in x.chunk(256, dim=1):
        chnked_out.append(sse.forward_mbc(chnk))
        out_mbc = chnked_out[-1]

    diff = (out - out_mbc).abs().amax()
    print(f"{out.size()=} {out_mbc.size()=}")
    print(f"max diff: {diff=}")

    for i in range(out_cumulative.size(1)):
        chunked = chnked_out[i]
        cumulative = out_cumulative[:, i]

        diff = (chunked - cumulative).abs().amax()
        print(f"max diff: {diff=}")

    cumulative16 = sse.forward_cumulative(x, chunk_size=16)
    cumulative32 = sse.forward_cumulative(x, chunk_size=32)
    cumulative16 = torch.stack(
        [cumulative16[:, i + (i + 1)] for i in range(256 // 32)], dim=1)

    for i in range(cumulative32.size(1)):
        s = cumulative16[:, i]
        t = cumulative32[:, i]
        diff = (s - t).abs().amax()
        print(f"max diff: {diff=}")
