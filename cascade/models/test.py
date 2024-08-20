from cascade.models.cascading_cache import SinkAttentionNaive, CascadingKVCache, CascadingKVCacheSlow, add_sinks, append_to_cache, evict_from_cache, overwrite_cache
import unittest
from argparse import Namespace
from cascade.main.llama_eval import load_model
import torch
import time
from cascade.models.cascade_attention import sample_monkeypatch


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
        )
        self.args = args

    def test_cascade_attention_iterative_generate(self):
        args = self.args
        for model in ["llama3.1-8b", "qwen7b"]:
            args.model = model
            model, tokenizer, device = load_model(args)
            model = sample_monkeypatch(model)
            past_key_values = get_cache(model)

            stride = args.cascade_stride
            inputs = torch.randint(0, 1024, size=(1, 2048), device=device)

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


class TestCascadingKVCache(unittest.TestCase):
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
