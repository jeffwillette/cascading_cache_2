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

    def __init__(self, K: int, h: int, slot_type: str) -> None:
        super().__init__()
        self.name = "Slots"
        self.K = K  # Number of Slots
        self.h = h  # Slot size
        self.slot_type = slot_type  # Deterministic or Random

        if slot_type not in ["random", "deterministic"]:
            raise ValueError("{} not implemented for slots".format(
                self.slot_type))

        if slot_type == "random":
            # same initialization as "Weight Uncertainty in Neural Networks"
            self.mu = nn.Parameter(torch.zeros(1, self.K,
                                               self.h).uniform_(-0.2, 0.2),
                                   requires_grad=True)
            self.sigma = nn.Parameter(torch.zeros(1, self.K,
                                                  self.h).uniform_(-5.0, -4.0),
                                      requires_grad=True)
            return

        self.S = nn.Parameter(torch.zeros(1, self.K, self.h),
                              requires_grad=True)
        nn.init.xavier_uniform_(self.S)  # type: ignore

    def sample_s(self) -> T:
        if self.slot_type == "random":
            if self.training:
                return (  # type: ignore
                    torch.randn_like(self.mu) * \
                    F.softplus(self.sigma) + self.mu
                )
            return self.mu

        # returning the parameter S caused problems because it is an
        # nn.Parameter and the above random returns a tensor. Return
        # a tensor here to make it consistent
        return (  # type: ignore
            torch.ones_like(self.S) * self.S)


class SlotSetEncoder(MBCFunction):

    def __init__(
        self,
        K: int,
        dim: int,
        slot_type: str = "random",
        ln_slots: bool = True,
        heads: int = 4,
        bias: bool = True,
        slot_drop: float = 0.0,
        attn_act: str = "slot-sigmoid",
        slot_residual: bool = True,
        eps: float = EPS,
        ln_after: bool = True,
    ):
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
        self.slot_residual = slot_residual

        self.slots: Slots

        if dim % heads != 0:
            raise ValueError(
                f"for multihead attention, {dim} must be evenly divisible by {heads}"
            )  # noqa

        # Normalization Term
        self.sqrt_dim = 1.0 / math.sqrt(dim // heads)
        self.slots = Slots(K=K, h=dim, slot_type=slot_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

        self.norm_slots = nn.LayerNorm(
            normalized_shape=dim) if ln_slots else nn.Identity()

        self.pre_sampled_slots = T()
        self.norm_after = nn.LayerNorm(dim) if ln_after else nn.Identity()
        self.pre_sampled = False

    def sample_s(self) -> T:
        S = self.slots.sample_s()
        S = self.norm_slots(S)

        if self.slot_drop <= 0.0 or not self.training:
            return self.q(S)  # type: ignore

        idx = torch.rand(self.slots.K) > self.slot_drop
        # we need to ensure that at least one slot is not dropped
        if idx.sum() == 0:
            lucky_one = torch.randperm(self.slots.K)[0]
            idx[lucky_one] = True

        return self.q(S[:, idx])  # type: ignore

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

        Q, K, V = S, self.k(X), self.v(X)  # (B, T, D)
        Q, K, V = map(self.head_split, (Q, K, V))  # (B, H, T, D)

        A = torch.einsum("bhkd,bhsd->bhks", Q * np.sqrt(self.sqrt_dim),
                         K * np.sqrt(self.sqrt_dim))
        return Q, A, V

    def forward(self, X: T, S: OT = None, mask: OT = None) -> T:
        B, T, D = X.size()
        if self.pre_sampled:
            S = self.pre_sampled_slots

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


if __name__ == "__main__":
    sse = SlotSetEncoder(
        K=16,
        dim=128,
        slot_type="deterministic",
        bias=False,
    )

    out = sse(torch.randn(2, 256, 128))
