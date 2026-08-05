"""Peak-memory benchmark: measure GPU HBM saved by activation offloading.

On CUDA: reports ``torch.cuda.max_memory_allocated`` for baseline vs. offload,
plus step time, so you can read the actual HBM reduction and the overlap cost.
On CPU: no device memory to free, so it reports the byte-accounting (how much
*would* be moved off a GPU) as a proxy, and validates the code path runs.

Usage:
    python benchmark/bench_memory.py --layers 24 --dmodel 2048 --seq 4096 --batch 1
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from act_offload import offload_activations  # noqa: E402
from act_offload.toy import TransformerStack  # noqa: E402


def step(model, x, offload, min_bytes, emulate):
    mgr = None
    model.zero_grad(set_to_none=True)
    if offload:
        cm = offload_activations(model, min_bytes=min_bytes, emulate_cpu=emulate)
    else:
        import contextlib

        cm = contextlib.nullcontext()
    with cm as mgr:
        out = model(x)
        loss = out.float().pow(2).mean()
        loss.backward()
    return mgr


def measure(model, x, offload, min_bytes, emulate, iters=3):
    cuda = x.is_cuda
    # warmup
    step(model, x, offload, min_bytes, emulate)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    mgr = None
    for _ in range(iters):
        mgr = step(model, x, offload, min_bytes, emulate)
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() if cuda else None
    return peak, dt, mgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dmodel", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dff", type=int, default=4096)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--min-bytes", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulate = device == "cpu"
    torch.manual_seed(0)

    model = TransformerStack(args.layers, args.dmodel, args.heads, args.dff).to(device)
    x = torch.randn(args.batch, args.seq, args.dmodel, device=device)

    mb = 1024 * 1024
    print(f"device={device}  layers={args.layers} d_model={args.dmodel} "
          f"seq={args.seq} batch={args.batch} min_bytes={args.min_bytes}")

    base_peak, base_dt, _ = measure(model, x, False, args.min_bytes, emulate, args.iters)
    off_peak, off_dt, mgr = measure(model, x, True, args.min_bytes, emulate, args.iters)

    if device == "cuda":
        print(f"  baseline : peak HBM = {base_peak / mb:8.1f} MiB   step = {base_dt*1e3:7.1f} ms")
        print(f"  offload  : peak HBM = {off_peak / mb:8.1f} MiB   step = {off_dt*1e3:7.1f} ms")
        saved = (base_peak - off_peak) / mb
        print(f"  --> saved {saved:.1f} MiB "
              f"({100*(base_peak-off_peak)/base_peak:.1f}%)  "
              f"time overhead {100*(off_dt-base_dt)/base_dt:+.1f}%")
        print("  " + mgr.summary())
    else:
        print("  (CPU: no device HBM to free; reporting offload byte-accounting)")
        print("  baseline step = {:.1f} ms".format(base_dt * 1e3))
        print("  offload  step = {:.1f} ms".format(off_dt * 1e3))
        print("  " + mgr.summary())
        print("  --> on a GPU this many bytes would be evicted from HBM during forward.")


if __name__ == "__main__":
    main()
