from cascade.models.cascading_cache import SinkAttentionNaive, CascadingKVCache, CascadingKVCacheSlow, add_sinks, append_to_cache, evict_from_cache, overwrite_cache
import numpy as np
import unittest
import triton

from argparse import Namespace
from cascade.main.llama_eval import load_model
import torch
import time
from cascade.models.cascade_attention import sample_monkeypatch
from cascade.models.flash_attention import attention


def get_cache(model):
    mdl = model.model
    window = mdl.config._window // mdl.config._cascades
    max_seq_len = mdl.config._window

    return CascadingKVCache(
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
        eager_fill=False,
    )


class TestCascadeAttention(unittest.TestCase):
    def setUp(self):
        args = Namespace(
            batch_size=1,
            sinks=4,
            cascades=4,
            window=2048,
            world_size=1,
            cascade_func="pow2",
            head_reduction="mean",
            method="sink",
            cascade_stride=128,
            job="test",
            lora_r=0,
            homogeneous_heads=True,
            do_og_pos=False,
        )
        self.args = args
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(True)

    def test_cascade_attention_iterative_generate(self):
        args = self.args
        for model in ["llama3.1-8b", "qwen7b"]:
            args.model = model
            model, tokenizer, device = load_model(args)
            model = sample_monkeypatch(model)
            past_key_values = get_cache(model)

            stride = args.cascade_stride
            inputs = torch.randint(0, 1024, size=(args.batch_size, 2048), device=device)

            # test a manual iterative generation
            for i in range(0, inputs.size(1), stride):
                inp = inputs[:, i: i + args.cascade_stride]
                output = model(inp, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                del output

            past_key_values.reset(verbose=True)

            # test the genreate method monkeypatch
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

            _ = tokenizer.decode(output[inputs.size(1):])

    def test_cascade_attention_with_cache_equals_dense(self):
        """
        with a seqeuence length which gits inside the sliding window, it should
        be equivalent to dense attention.
        """
        args = self.args
        args.cascades = 1
        args.model = "llama3.1-8b-instruct"

        model, tokenizer, device = load_model(args)
        model = sample_monkeypatch(model)
        past_key_values = get_cache(model)

        stride = args.cascade_stride
        inputs = torch.randint(0, 1024, size=(args.batch_size, 1024 - (128 - 92)), device=device)

        # test a manual iterative generation
        cascade_logits = []
        for i in range(0, inputs.size(1), stride):
            inp = inputs[:, i: i + args.cascade_stride]
            output = model(inp, past_key_values=past_key_values, use_cache=True)
            cascade_logits += [output.logits]
            past_key_values = output.past_key_values
            del output

        past_key_values.reset(verbose=True)
        output = model(inputs, past_key_values=past_key_values, use_cache=True)
        dense_logits = output.logits
        dense_logits = torch.split(dense_logits, args.cascade_stride, dim=1)

        for c, d in zip(cascade_logits, dense_logits):
            diff = (c - d).abs().mean(dim=-1)
            print(f"{diff=}")
            argmax_same = c.argmax(dim=-1) == d.argmax(dim=-1)
            print(f"{argmax_same=}")


class TestFlashAttention(unittest.TestCase):
    def test_flash_attention_fwd_bwd(self):
        def test_op_with_backward(Z, H, N_CTX, N_KV, HEAD_DIM, causal, dtype=torch.float16):
            torch.manual_seed(20)
            q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
            k = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
            v = (torch.empty((Z, H, N_KV, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
            M = torch.full((N_CTX, N_KV), 1, dtype=torch.bool, device="cuda").triu(1)

            sm_scale = 0.5
            dout = torch.randn_like(q)
            # reference implementation
            p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if causal:
                p[:, :, M == 1] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).half()
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v)
            ref_out.backward(dout)
            ref_dv, v.grad = v.grad.clone(), None
            ref_dk, k.grad = k.grad.clone(), None
            ref_dq, q.grad = q.grad.clone(), None
            # triton implementation
            tri_out, _ = attention(q, k, v, causal, sm_scale, M, 1.0)
            tri_out = tri_out.half()

            # compare
            close = torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
            self.assertTrue(close)

            # fwd_diff = (ref_out - tri_out).abs().mean().amax()
            # print(f"{fwd_diff=}")

            tri_out.backward(dout, None)
            tri_dv, v.grad = v.grad.clone(), None
            tri_dk, k.grad = k.grad.clone(), None
            tri_dq, q.grad = q.grad.clone(), None

            rtol = 0.0
            # Relative tolerance workaround for known hardware limitation of MI200 GPU.
            # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
            if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
                rtol = 1e-2

            dv_close = torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
            dk_close = torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
            dq_close = torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)

            # dv_diff = (ref_dv - tri_dv).abs()
            # dk_diff = (ref_dk - tri_dk).abs()
            # dq_diff = (ref_dq - tri_dq).abs()
            # print(f"{dv_diff=} {dk_diff=} {dq_diff=}")

            # print(f"{dv_diff=}")
            # print(f"{dk_diff=}")
            # print(f"{dq_diff=}")

            self.assertTrue(dv_close)
            self.assertTrue(dk_close)
            self.assertTrue(dq_close)

        for Z, H, N_CTX, N_KV, HEAD_DIM in (
            (1, 2, 128, 128, 64),
            (1, 2, 1024, 16384 + 1024 + 4, 64),
            (2, 32, 1024, 16384 + 1024 + 4, 128),
        ):
            test_op_with_backward(Z, H, N_CTX, N_KV, HEAD_DIM, True)


class TestCascadingKVCache(unittest.TestCase):
    def test_different_sized_layer_caches(self):
        # toy settings
        cache_sizes = [2**i for i in range(5, 10)]
        windows = [2**i for i in range(4)]

        for i in range(10):
            n_layers = np.random.randint(8, 33, size=(1,)).tolist()[0]
            N = 1
            HID = 128
            NSINK = 4
            HEAD = 1
            DEVICE = "cuda:0"
            DTYPE = torch.float16

            MAX_SEQ = np.random.choice(np.array(cache_sizes), size=(n_layers,)).tolist()
            w = np.random.choice(windows, size=(n_layers,)).tolist()

            # MAX_SEQ = [s for _ in range(n_layers)]
            WIND = [ms // win for ms, win in zip(MAX_SEQ, w)]

            S = torch.randint(max(MAX_SEQ) + 4, max(MAX_SEQ) * 2, size=(1,)).item()
            # S = s + 4
            with torch.no_grad():

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=n_layers,
                    eager_fill=True,
                )

                for l in range(n_layers):
                    _k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                    _v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                    k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                        cache.update(_k, _v, l)

                    self.assertEqual(k_nosink.size(2), MAX_SEQ[l], f"{k.size()=} expected to match {MAX_SEQ[l]=} at dim 2")
                    self.assertEqual(v_nosink.size(2), MAX_SEQ[l], f"{v.size()=} expected to match {MAX_SEQ[l]=} at dim 2")
                    self.assertEqual(mask.sum().item(), 0,
                                     "expected nothing to be masked due to eager filling: " + \
                                     f"({MAX_SEQ[l]=}, {k_nosink.size()=}, {mask.size()=}, queries: {_k.size()=})\n{mask=}" + \
                                     f"{WIND[l]=} layer: {l=} {cache.stored_tokens[l]=}"
                                     )

    def test_against_naive_single_iter(self):
        # toy settings
        N = 1
        HID = 128
        NSINK = 4
        WIND = 4
        HEAD = 32
        MAX_SEQ = 32
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        for i in range(10):
            with torch.no_grad():
                S = torch.randint(200, 500, size=(1,)).item()

                cache_slow = CascadingKVCacheSlow(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    n_layers=1,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                cache_slow.add_sinks = add_sinks
                cache_slow.append_to_cache = append_to_cache
                cache_slow.evict_from_cache = evict_from_cache
                cache_slow.overwrite_cache = overwrite_cache

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=False,
                )

                slow_times, fast_times = [], []

                # k = torch.arange(S, device=DEVICE, dtype=DTYPE) + 1
                # k = k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                # v = k.clone()
                _k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                _v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                for i in range(_k.size(2)):
                    # print(f"\n\n{'='*100}\n\n")
                    tic = time.perf_counter()
                    (
                        k_slow,
                        v_slow,
                        pos_slow,
                        sink_mask_slow,
                        k_nosink_slow,
                        v_nosink_slow,
                        pos_nosink_slow,
                        mask_slow,
                    ) = cache_slow.update(_k[:, :, i:i + 1].clone(), _v[:, :, i:i + 1].clone())

                    slow_times.append(time.perf_counter() - tic)

                    # print(f"{k=}\n{v=}\n{pos=}\n{sink_mask=}")
                    k_slow, v_slow = torch.cat((k_slow, k_nosink_slow),
                                               dim=-2), torch.cat(
                        (v_slow, v_nosink_slow), dim=-2)
                    pos_slow = torch.cat((pos_slow, pos_nosink_slow),
                                         dim=-1).squeeze(0)

                    # print(f"{k[0, 0, :,  0]=}")
                    mask = torch.cat((sink_mask_slow, mask_slow), dim=-1)

                    n = (mask == 0).sum()
                    k_slow, v_slow = k_slow[:, :, :n], v_slow[:, :, :n]
                    argsort = torch.argsort(pos_slow[:n])
                    # print(
                    #     f"before sort slow: \n{k_slow.reshape(-1)=}\n{pos_slow.reshape(-1)=}"
                    # )

                    # print(f"{k_nocomp[0, 0, :, 0]=}")
                    k_slow, v_slow = k_slow[:, :, argsort], v_slow[:, :, argsort]

                    tic = time.perf_counter()
                    k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                        cache.update(_k[:, :, i:i + 1], _v[:, :, i:i + 1], 0)

                    fast_times.append(time.perf_counter() - tic)

                    # print(
                    #     f"fast from mdoel out {pos.size()=} {pos_nosink.size()=} {k.size()=} {k_nosink.size()=}"
                    # )
                    # print(f"sink: {k}")
                    # print(f"nosink: {k_nosink}")
                    # print(f"og pos: {cache.og_pos[0, 0]=}")
                    idx = 0
                    pos, mask, sink_mask, pos_nosink = pos[idx], mask[
                        idx, 0], sink_mask[idx, 0], pos_nosink[idx]

                    # print(f"{k.size()=}")
                    # print(f"{k=}")
                    # print(f"{k[idx:idx+1].view(-1)=}\n{pos=}\n{sink_mask=}")
                    k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink), dim=-2)
                    pos = torch.cat((pos, pos_nosink), dim=-1)

                    # print(f"{k[0, 0, :,  0]=}")
                    mask = torch.cat((sink_mask, mask), dim=-1)

                    n = (mask != 1).sum()
                    k, v = k[idx:idx + 1, :, :n], v[idx:idx + 1, :, :n]
                    argsort = torch.argsort(pos[:n])
                    # print(
                    #     f"fast: {mask=}\n{k[:, :].reshape(-1)=}\n{v[:, :].reshape(-1)=}"
                    # )

                    # print(f"before sort: \n{k.reshape(-1)=}\n{pos.reshape(-1)=}")

                    k, v = k[:, :, argsort], v[:, :, argsort]

                    # print(f"after sort: {k.view(-1)}")
                    if not k_slow.size() == k.size():
                        print(f"{k_slow.size()=} {k.size()=}")
                        print(f"sizes not equal...\n{k_slow=} {k=}")
                        exit()

                    diff = (k_slow - k).abs().sum()
                    pos_diff = (pos - pos_slow).abs()
                    # torch.set_printoptions(profile="default")
                    if diff > 1e-6 or pos_diff.sum() > 0:
                        print(f"{k_slow=}")
                        print(f"{k_slow[0, 0, -100:, 0]=}")
                        print(f"{k=}")
                        print(f"{(k - k_slow).abs()=}")
                        print(
                            f"{k_slow.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_slow).abs().view(-1)=}"
                        )
                        print(f"pos diff: {(pos - pos_slow).abs()} {(pos - pos_slow).sum()=}")
                        self.fail("failed difference test")

            slow_times = slow_times[100:]
            fast_times = fast_times[100:]
            slow_times = sum(slow_times) / len(slow_times)
            fast_times = sum(fast_times) / len(fast_times)
            print(f"{slow_times=} {fast_times=} ratio {slow_times / fast_times}")

    def test_against_naive_batched(self):
        # toy settings
        N = 1
        HID = 128
        NSINK = 4
        WIND = 512
        HEAD = 32
        MAX_SEQ = 2048
        DEVICE = "cuda:0"
        DTYPE = torch.float16

        for i in range(10):
            with torch.no_grad():
                S = torch.randint(1000, 20000, size=(1,)).item()

                cache_slow = CascadingKVCacheSlow(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    n_layers=1,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                cache_slow.add_sinks = add_sinks
                cache_slow.append_to_cache = append_to_cache
                cache_slow.evict_from_cache = evict_from_cache
                cache_slow.overwrite_cache = overwrite_cache

                cache = CascadingKVCache(
                    window_length=WIND,
                    num_sink_tokens=NSINK,
                    max_batch_size=N,
                    heads=HEAD,
                    dim=HID,
                    max_seq_len=MAX_SEQ,
                    device=DEVICE,
                    dtype=DTYPE,
                    layers=1,
                    eager_fill=False,
                )

                slow_times, fast_times = [], []

                # k = torch.arange(S, device=DEVICE, dtype=DTYPE) + 1
                # k = k.view(1, 1, -1, 1).repeat(N, HEAD, 1, HID)
                # v = k.clone()
                k = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)
                v = torch.randn(N, HEAD, S, HID, device=DEVICE, dtype=DTYPE)

                for i in range(k.size(2)):
                    # print(f"\n\n{'='*100}\n\n")
                    tic = time.perf_counter()
                    (
                        k_slow,
                        v_slow,
                        pos_slow,
                        sink_mask_slow,
                        k_nosink_slow,
                        v_nosink_slow,
                        pos_nosink_slow,
                        mask_slow,
                    ) = cache_slow.update(k[:, :, i:i + 1].clone(), v[:, :, i:i + 1].clone())

                    slow_times.append(time.perf_counter() - tic)

                    # print(f"{k=}\n{v=}\n{pos=}\n{sink_mask=}")
                    k_slow, v_slow = torch.cat((k_slow, k_nosink_slow),
                                               dim=-2), torch.cat(
                        (v_slow, v_nosink_slow), dim=-2)
                    pos_slow = torch.cat((pos_slow, pos_nosink_slow),
                                         dim=-1).squeeze(0)

                    # print(f"{k[0, 0, :,  0]=}")
                    mask = torch.cat((sink_mask_slow, mask_slow), dim=-1)

                    n = (mask == 0).sum()
                    k_slow, v_slow = k_slow[:, :, :n], v_slow[:, :, :n]
                    argsort = torch.argsort(pos_slow[:n])
                    # print(
                    #     f"before sort slow: \n{k_slow.reshape(-1)=}\n{pos_slow.reshape(-1)=}"
                    # )

                    # print(f"{k_nocomp[0, 0, :, 0]=}")
                    k_slow, v_slow = k_slow[:, :, argsort], v_slow[:, :, argsort]

                # ============================================================================================
                _ = cache.update(k, v, 0)
                cache.reset()

                tic = time.perf_counter()
                k, v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, of_pos = \
                    cache.update(k, v, 0)

                fast_times.append(time.perf_counter() - tic)

                # print(
                #     f"fast from mdoel out {pos.size()=} {pos_nosink.size()=} {k.size()=} {k_nosink.size()=}"
                # )
                # print(f"sink: {k}")
                # print(f"nosink: {k_nosink}")
                # print(f"og pos: {cache.og_pos[0, 0]=}")
                idx = 0
                pos, mask, sink_mask, pos_nosink = pos[idx], mask[
                    idx, 0], sink_mask[idx, 0], pos_nosink[idx]

                # print(f"{k.size()=}")
                # print(f"{k=}")
                # print(f"{k[idx:idx+1].view(-1)=}\n{pos=}\n{sink_mask=}")
                k, v = torch.cat((k, k_nosink), dim=-2), torch.cat((v, v_nosink), dim=-2)
                pos = torch.cat((pos, pos_nosink), dim=-1)

                # print(f"{k[0, 0, :,  0]=}")
                mask = torch.cat((sink_mask, mask), dim=-1)

                n = (mask != 1).sum()
                k, v = k[idx:idx + 1, :, :n], v[idx:idx + 1, :, :n]
                argsort = torch.argsort(pos[:n])
                # print(
                #     f"fast: {mask=}\n{k[:, :].reshape(-1)=}\n{v[:, :].reshape(-1)=}"
                # )

                # print(f"before sort: \n{k.reshape(-1)=}\n{pos.reshape(-1)=}")

                k, v = k[:, :, argsort], v[:, :, argsort]

                # print(f"after sort: {k.view(-1)}")
                if not k_slow.size() == k.size():
                    print(f"{k_slow.size()=} {k.size()=}")
                    print(f"sizes not equal...\n{k_slow=} {k=}")
                    exit()

                diff = (k_slow - k).abs().sum()
                if diff > 1e-6:
                    print(f"{k_slow=}")
                    print(f"{k_slow[0, 0, -100:, 0]=}")
                    print(f"{k=}")
                    print(f"{(k - k_slow).abs()=}")
                    print(
                        f"{k_slow.view(-1)=}\n{k.view(-1)=}\n{pos.view(-1)=}\n{(k - k_slow).abs().view(-1)=}"
                    )
                    print(f"pos diff: {(pos - pos_slow).abs()} {(pos - pos_slow).sum()=}")
                    exit("too big")

            slow_times = slow_times[100:]
            slow_times = sum(slow_times) / len(slow_times)
            fast_times = sum(fast_times) / len(fast_times) / S
            print(f"{slow_times=} {fast_times=} ratio {slow_times / fast_times}")

    def test_bench_against_naive(self):
        naive = SinkAttentionNaive(4, 1024).cuda()

        dev = "cuda:0"
        cache = CascadingKVCache(
            window_length=1024 // 1,
            num_sink_tokens=4,
            max_batch_size=8,
            heads=32,
            dim=128,
            max_seq_len=1024,
            device=dev,
            dtype=torch.float32,
            layers=1,
        )

        cache_cascade = CascadingKVCache(
            window_length=1024 // 4,
            num_sink_tokens=4,
            max_batch_size=8,
            heads=32,
            dim=128,
            max_seq_len=1024,
            device=dev,
            dtype=torch.float32,
            layers=1,
            eager_fill=False,
        )

        naive_times, fast_times, cascade_times = [], [], []
        its = 4096
        # testing the original version
        for i in range(its + 100):
            k = torch.randn(8, 32, 1, 128).cuda()
            v = torch.randn(8, 32, 1, 128).cuda()
            attn_score = torch.randn(8, 32, 1024).cuda()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            sink_k, sink_v, k_cache, v_cache = naive(k, v)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            naive_times.append(elapsed)

        k = torch.randn(8, 32, its, 128).cuda()
        v = torch.randn(8, 32, its, 128).cuda()
        attn_score = torch.randn(8, 32, its).cuda()

        # testing the cascade fast version
        # burn in period for triton compile
        _ = cache_cascade.update(k, v, 0, attn_score)
        cache_cascade.reset()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _k, _v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, og_pos = cache_cascade.update(
            k, v, 0, attn_score)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        cascade_times.append(elapsed)

        # testing the fast version of SLLM
        # burn in period for triton compile
        _ = cache.update(k, v, 0, attn_score)
        cache.reset()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _k, _v, pos, sink_mask, k_nosink, v_nosink, pos_nosink, mask, og_pos = cache.update(
            k, v, 0, attn_score)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        fast_times.append(elapsed)

        naive_mean = sum(naive_times[100:]) / len(naive_times[100:])
        fast_mean = sum(fast_times) / len(fast_times) / its
        cascade_mean = sum(cascade_times) / len(cascade_times) / its
        print(f"{naive_mean=} {fast_mean=} {cascade_mean=}")
        print(f"{cascade_mean / naive_mean=}")
        print(f"{fast_mean / naive_mean=}")


if __name__ == "__main__":
    N = 128
    HID = 128
    NSINK = 4
    WIND = 2048 // 4
    HEAD = 32
    MAX_SEQ = 2048
    DEVICE = "cuda:3"
    DTYPE = torch.float16

    # for nsys profiling
    cache = CascadingKVCache(
        window_length=WIND,
        num_sink_tokens=NSINK,
        max_batch_size=N,
        heads=HEAD,
        dim=HID,
        max_seq_len=MAX_SEQ,
        device=DEVICE,
        dtype=DTYPE,
        layers=1,
        eager_fill=False,
    )

    ITS = 16384
    with torch.no_grad():
        s_in = torch.randn(N, HEAD, ITS, device=DEVICE, dtype=DTYPE)
        k_in = torch.randn(N, HEAD, ITS, HID, device=DEVICE, dtype=DTYPE)
        s_in = torch.randn(N, HEAD, ITS, device=DEVICE, dtype=DTYPE)
        v_in = torch.randn(N, HEAD, ITS, HID, device=DEVICE, dtype=DTYPE)

        for k_i, v_i, s_i in zip(k_in.chunk(1, dim=2), v_in.chunk(1, dim=2),
                                 s_in.chunk(1, dim=2)):
            _ = cache.update_batch(k_i, v_i, s_i, 0, )
