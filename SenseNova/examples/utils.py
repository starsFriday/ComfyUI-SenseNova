from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ..layer_streaming import SimpleLayerStreamingWrapper


_M = TypeVar("_M", bound=torch.nn.Module)


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@contextmanager
def _streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    wrapper = SimpleLayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        active_count=prefetch_count,
    )
    try:
        yield wrapper
    finally:
        wrapper.teardown()
        cleanup_memory()
        torch.cuda.synchronize(device=target_device)
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
