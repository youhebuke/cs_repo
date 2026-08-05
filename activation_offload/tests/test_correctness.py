"""Numerical-correctness test: offloading saved activations must not change results.

Runs on CPU (via emulate_cpu) or GPU. Verifies that forward outputs and *all*
parameter gradients are identical (within fp tolerance) with vs. without the
activation-offload hooks installed.
"""
import copy
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from act_offload import offload_activations  # noqa: E402
from act_offload.toy import TransformerStack  # noqa: E402


def _run(model, x):
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return loss.detach().clone(), out.detach().clone(), grads


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulate = device == "cpu"

    model = TransformerStack(n_layers=4, d_model=256, n_heads=8, d_ff=1024).to(device)
    ref = copy.deepcopy(model)
    x = torch.randn(2, 128, 256, device=device)

    # baseline (no offload)
    loss0, out0, g0 = _run(ref, x)

    # with offload (small threshold so activations actually get offloaded)
    mgr = None
    with offload_activations(model, min_bytes=4096, emulate_cpu=emulate) as mgr:
        loss1, out1, g1 = _run(model, x)

    assert torch.allclose(loss0, loss1, atol=1e-5, rtol=1e-4), (loss0, loss1)
    assert torch.allclose(out0, out1, atol=1e-5, rtol=1e-4)
    max_gd = 0.0
    for n in g0:
        d = (g0[n] - g1[n]).abs().max().item()
        max_gd = max(max_gd, d)
        assert torch.allclose(g0[n], g1[n], atol=1e-5, rtol=1e-4), (n, d)

    assert mgr.stats["offloaded_count"] > 0, "nothing was offloaded -- threshold too high?"
    print(f"[OK] device={device} emulate_cpu={emulate}")
    print(f"     loss baseline={loss0.item():.6f} offload={loss1.item():.6f}")
    print(f"     max |grad diff| = {max_gd:.3e}")
    print("     " + mgr.summary())


def test_offload_correctness():
    """pytest entry point."""
    main()


if __name__ == "__main__":
    main()
