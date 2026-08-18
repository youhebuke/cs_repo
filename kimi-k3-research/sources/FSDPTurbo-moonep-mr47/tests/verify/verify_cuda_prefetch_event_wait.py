#!/usr/bin/env python3
"""Verify CUDA event waits never go through Event.wait(accelerator stream).

Run from the repository root:

    PYTHONPATH=. python3 tests/verify/verify_cuda_prefetch_event_wait.py

Expected output (exit code 0):

    [1/4] PASS  CUDA 路径调用 stream.wait_event, 不调用 event.wait
    [2/4] PASS  event 为 None 时是空操作
    [3/4] PASS  传入没有 wait_event 的 stream 时回退到 torch.cuda.current_stream()
    [4/4] PASS  adapter 源码不再把 accelerator stream 传给 Event.wait
ALL PASS
"""

import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsdp_turbo.distributed.expert_parallel import moonep_adapter  # noqa: E402

_checks = []
_TOTAL = 4


def check(passed: bool, message: str) -> None:
    _checks.append((passed, message))
    print(f"[{len(_checks)}/{_TOTAL}] {'PASS' if passed else 'FAIL'}  {message}")


def check_cuda_uses_stream_wait_event():
    calls = []

    class FakeStream:
        def wait_event(self, event):
            calls.append(("stream.wait_event", event))

    class FakeEvent:
        def wait(self, stream):
            calls.append(("event.wait", stream))

    original = moonep_adapter._is_cuda
    moonep_adapter._is_cuda = lambda: True
    try:
        event = FakeEvent()
        moonep_adapter._wait_event(event, FakeStream())
        passed = calls == [("stream.wait_event", event)]
    finally:
        moonep_adapter._is_cuda = original
    check(passed, "CUDA 路径调用 stream.wait_event, 不调用 event.wait")


def check_none_is_noop():
    original = moonep_adapter._is_cuda
    moonep_adapter._is_cuda = lambda: True
    try:
        moonep_adapter._wait_event(None)
        passed = True
    except Exception:
        passed = False
    finally:
        moonep_adapter._is_cuda = original
    check(passed, "event 为 None 时是空操作")


def check_non_cuda_stream_falls_back_to_current():
    calls = []

    class CurrentStream:
        def wait_event(self, event):
            calls.append(event)

    class ForeignStream:
        pass

    class FakeEvent:
        def wait(self, stream):
            raise AssertionError("event.wait must not run on CUDA")

    original_is_cuda = moonep_adapter._is_cuda
    original_current = torch.cuda.current_stream
    moonep_adapter._is_cuda = lambda: True
    torch.cuda.current_stream = lambda: CurrentStream()
    try:
        event = FakeEvent()
        moonep_adapter._wait_event(event, ForeignStream())
        passed = calls == [event]
    finally:
        moonep_adapter._is_cuda = original_is_cuda
        torch.cuda.current_stream = original_current
    check(passed, "传入没有 wait_event 的 stream 时回退到 torch.cuda.current_stream()")


def check_source_does_not_call_event_wait_with_accelerator_stream():
    source = inspect.getsource(moonep_adapter.MoonEPCallState)
    banned = (
        "event.wait(torch.accelerator.current_stream())" in source
        or "done.wait(torch.accelerator.current_stream())" in source
        or "with torch.accelerator.stream(comm_stream)" in source
    )
    uses_helper = "_wait_event(" in source and "_stream_ctx(" in source
    check(
        uses_helper and not banned,
        "adapter 源码不再把 accelerator stream 传给 Event.wait",
    )


def main() -> int:
    check_cuda_uses_stream_wait_event()
    check_none_is_noop()
    check_non_cuda_stream_falls_back_to_current()
    check_source_does_not_call_event_wait_with_accelerator_stream()

    failures = [message for passed, message in _checks if not passed]
    if failures:
        print(f"\nFAILED {len(failures)} check(s)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
