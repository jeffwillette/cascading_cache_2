import pytest
import torch

import triton
import triton.language as tl


def is_hip():
    return False
    # return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(
    acc,  # accumulator for qk - exp(max)
    l_i,  # running normalization constant sum
    m_i,  # running maximum
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    MASK,
    stride_mm,
    stride_mn,
    SCORES,
    stride_sn,
    beta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_KV: tl.constexpr,
    fp8_v: tl.constexpr,
):
    lo, hi = 0, N_KV

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    Mask_block = offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        mask = tl.load(MASK + Mask_block).to(tl.int64)

        qk = tl.dot(q, k)
        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0)

        exps = tl.flip(tl.arange(0, BLOCK_M))[:, None]
        # do beta ** exps * (1 - beta)
        unmasked = tl.where(mask, 0, 1)
        exps = tl.exp2(exps.to(tl.float16) * tl.log2(beta))
        coeff = exps * (1 - beta) * unmasked
        score_offset = (start_n + tl.arange(0, BLOCK_N)).to(
            tl.int64) * stride_sn

        local_qk = tl.exp2(qk - tl.max(qk, 1)[:, None])
        local_qk = local_qk / tl.sum(local_qk, 1)[:, None]
        tl.atomic_add(SCORES + score_offset, val=tl.sum(local_qk * coeff, 0))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)

        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        Mask_block += BLOCK_N

    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX"])
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        MASK,
        SCORES,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        stride_mm,
        stride_mn,
        stride_sz,
        stride_sh,
        stride_sn,
        Z,
        H,
        N_CTX,  #
        N_KV,
        beta: tl.constexpr,
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        STAGE: tl.constexpr  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    score_offset = off_z.to(tl.int64) * stride_sz + off_h.to(
        tl.int64) * stride_sh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(0 if V.dtype.element_ty == tl.float8e5 else 1,
               1 if V.dtype.element_ty == tl.float8e5 else 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        MASK,
        stride_mm,
        stride_mn,
        SCORES + score_offset,
        stride_sn,
        beta,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        offs_m,
        offs_n,
        N_KV,
        V.dtype.element_ty == tl.float8e5,  #
    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, mask, beta):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[
            -1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert k.size(2) == v.size(2)
        N_KV = k.size(2)

        # pad the key values because the BLOCK_N is always expecting some multiple of BLOCK_N
        # we will have to mask out the unwanted values in the attention_inner
        b, h, s, d = k.shape
        N_SCORES = N_KV
        if N_KV % 32 != 0:
            n = 32 - (N_KV % 32)  # + 32
            k = torch.cat(
                (k, torch.zeros(b, h, n, d, dtype=k.dtype, device=k.device)),
                dim=2)
            v = torch.cat(
                (v, torch.zeros(b, h, n, d, device=v.device, dtype=v.dtype)),
                dim=2)

            mask = torch.cat(
                (mask,
                 torch.ones(
                     mask.size(0), n, dtype=mask.dtype, device=mask.device)),
                dim=-1)

            N_SCORES += n

        scores = torch.zeros(q.size(0),
                             q.size(1),
                             N_SCORES,
                             device=q.device,
                             dtype=q.dtype)

        og_q = q.size(2)
        if q.size(2) % 64 != 0:
            n = 64 - (q.size(2) % 64)
            q = torch.cat((q, torch.zeros(b, h, n, d, dtype=q.dtype, device=q.device)), dim=2)
            mask = torch.cat(
                (mask,
                 torch.ones(
                     n, mask.size(1), dtype=mask.dtype, device=mask.device)),
                dim=0)

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {
                "waves_per_eu": waves_per_eu,
                "allow_flush_denorm": True
            }

        def grid(args):
            return (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                        device=q.device,
                        dtype=torch.float32)

        # scores = torch.zeros(q.size(0),
        #                      q.size(1),
        #                      N_KV,
        #                      device=q.device,
        #                      dtype=q.dtype)

        # print(f"{q.size()=} {k.size()=} {v.size()=} {sm_scale=} {M.size()=} {o.size()=}")
        # print(f"{mask.size()=} {scores.size()=} {q.stride()=} {k.stride()=} {v.stride()=}")
        # print(f"{o.stride()=} {mask.stride()=} {scores.stride()=}")

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,  #
            mask,
            scores,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            mask.stride(0),
            mask.stride(1),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            q.shape[0],
            q.shape[1],  #
            N_CTX=q.shape[2],  #
            N_KV=N_KV,
            beta=beta,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            # temporary fix to hardcode
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=4,
            num_stages=3,
            **extra_kern_args)

        o = o[:, :, :og_q].contiguous()
        scores = scores[:, :, :N_KV]

        # print(f"{o.size()=} {scores.size()=}")

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

        return o, scores

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("refer to triton tutorials to get the bwd functions and modify them")


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16, N_KV=None):
    if N_KV is None:
        N_KV = N_CTX

    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype,
                     device="cuda").normal_(mean=0.0,
                                            std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    # M = torch.rand(N_CTX, N_KV, device="cuda") > 0.5
    M = ~torch.ones(N_CTX, N_KV, device="cuda", dtype=torch.bool).tril()
    print(f"{M} {M.size()=}")
    print(f"{(~M).sum(dim=0)=}")
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        p[:, :, M == 1] = float("-inf")
        a = torch.matmul(q, k.transpose(2, 3))
        a[:, :, M == 1] = 0
        print(f"analytic scores {a[0, 0].sum(dim=0)}")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # ref_out.backward(dout)
    # ref_dv, v.grad = v.grad.clone(), None
    # ref_dk, k.grad = k.grad.clone(), None
    # ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out, scores = attention(q, k, v, causal, sm_scale, M)
    tri_out = tri_out.half()
    # print(f"{(tri_attention - att).abs().amax()=}")
    # tri_out.backward(dout)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # compare
    diff = (ref_out - tri_out).abs()
    print(f"{diff.amax()=}")

    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # rtol = 0.0
    # # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # if torch.version.hip is not None and triton.runtime.driver.active.get_current_target(
    # ).arch == "gfx90a":
    #     rtol = 1e-2
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    # turned off due to GPU requirements, needed to test flash attention standalone
    # HAS_FLASH = True
    HAS_FLASH = False
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(9, 10)],
                line_arg="provider",
                line_vals=["triton-fp16"] +
                (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] +
                (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-")],
                ylabel="ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH,
                          H,
                          N_CTX,
                          HEAD_DIM,
                          causal,
                          mode,
                          provider,
                          device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM),
                        dtype=dtype,
                        device=device,
                        requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        def fn(): return attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            def fn(): return o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM),
                          dtype=dtype,
                          device=device,
                          requires_grad=True)

        def fn(): return flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            def fn(): return o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # print("64 test")
    # test_op(1, 1, 64, 64, True)

    # print("128 test")
    # test_op(10, 32, 128, 128, True, N_KV=128)
    print("32 test")
    test_op(1, 2, 64, 64, True, N_KV=32)
    exit()

    print("\n\n512/512\n\n")
    test_op(16, 32, 512, 128, True)

    print(f"\n\n512/64\n\n")
    test_op(16, 32, 512, 128, True, N_KV=64)
    print(f"\n\n512/2048\n\n")
    test_op(16, 32, 512, 128, True, N_KV=2048)
    print(f"\n\n512/2048+32\n\n")
    test_op(16, 32, 512, 128, True, N_KV=2048 + 32)
    print(f"\n\n512/2048+32+512\n\n")
    test_op(1, 32, 512, 128, True, N_KV=2048 + 32 + 512)
