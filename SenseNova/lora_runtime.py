from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open


_DOWN_SUFFIX = ".lora_down.weight"
_UP_SUFFIX = ".lora_up.weight"
_ALPHA_SUFFIX = ".alpha"
_BUFFER_DOWN = "_sensenova_lora_down"
_BUFFER_UP = "_sensenova_lora_up"


@dataclass(frozen=True)
class LoRATarget:
    down: torch.Tensor
    up: torch.Tensor
    scale: float


class SenseNovaLoRA:
    def __init__(self, path: str, metadata: dict[str, str], targets: dict[str, LoRATarget], strength: float) -> None:
        self.path = path
        self.metadata = metadata
        self.targets = targets
        self.strength = float(strength)

        source = metadata.get("source", "").lower()
        name = os.path.basename(path).lower()
        steps = metadata.get("comfyui_sensenova_steps")
        self.steps = int(steps) if steps is not None else 8 if "8step" in name or "dmd" in source else None
        self.task = metadata.get("comfyui_sensenova_task", "t2i" if self.steps is not None else "")
        self.cfg = float(metadata.get("comfyui_sensenova_cfg", "1.0")) if self.steps is not None else None
        self.cfg_norm = metadata.get("comfyui_sensenova_cfg_norm", "none") if self.steps is not None else None
        self.timestep_shift = float(metadata.get("comfyui_sensenova_timestep_shift", "3.0")) if self.steps is not None else None

    def validate_model(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = [name for name in self.targets if name not in modules]
        if missing:
            raise ValueError(f"SenseNova LoRA has {len(missing)} targets missing from this model: {', '.join(missing[:5])}")

        for name, target in self.targets.items():
            module = modules[name]
            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            if target.down.ndim != 2 or target.up.ndim != 2:
                raise ValueError(f"SenseNova LoRA target {name} is not a pair of matrices.")
            rank, target_in = target.down.shape
            target_out, up_rank = target.up.shape
            if rank != up_rank or target_in != in_features or target_out != out_features:
                raise ValueError(
                    f"SenseNova LoRA target {name} has shapes {tuple(target.down.shape)} and {tuple(target.up.shape)}, "
                    f"but the model layer is {in_features} -> {out_features}."
                )

    @contextmanager
    def apply(self, model: torch.nn.Module, dtype: torch.dtype) -> Iterator[None]:
        self.validate_model(model)
        modules = dict(model.named_modules())
        attached: list[torch.nn.Module] = []
        handles = []
        try:
            for name, target in self.targets.items():
                module = modules[name]
                if hasattr(module, _BUFFER_DOWN) or hasattr(module, _BUFFER_UP):
                    raise RuntimeError(f"SenseNova LoRA is already attached to {name}.")
                device = _module_device(module)
                module.register_buffer(_BUFFER_DOWN, target.down.to(device=device, dtype=dtype), persistent=False)
                module.register_buffer(_BUFFER_UP, target.up.to(device=device, dtype=dtype), persistent=False)
                attached.append(module)
                handles.append(module.register_forward_hook(functools.partial(_lora_forward_hook, scale=target.scale * self.strength)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            for module in attached:
                if hasattr(module, _BUFFER_DOWN):
                    delattr(module, _BUFFER_DOWN)
                if hasattr(module, _BUFFER_UP):
                    delattr(module, _BUFFER_UP)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor is not None:
            return tensor.device
    raise ValueError(f"Cannot determine device for LoRA target {type(module).__name__}.")


def _lora_forward_hook(module: torch.nn.Module, inputs, output: torch.Tensor, scale: float) -> torch.Tensor:
    hidden_states = inputs[0]
    down = getattr(module, _BUFFER_DOWN)
    up = getattr(module, _BUFFER_UP)
    residual = F.linear(F.linear(hidden_states, down), up)
    return torch.add(output, residual, alpha=scale)


def load_sensenova_lora(path: str, strength: float = 1.0) -> SenseNovaLoRA:
    if not path.lower().endswith(".safetensors"):
        raise ValueError("SenseNova U1.5 LoRA Loader only accepts .safetensors files.")

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        tensor_kind = metadata.get("tensor_kind")
        if tensor_kind not in (None, "neo_hf_lora"):
            raise ValueError(f"Unsupported SenseNova LoRA tensor_kind: {tensor_kind}.")

        keys = set(checkpoint.keys())
        down_keys = sorted(key for key in keys if key.endswith(_DOWN_SUFFIX))
        if not down_keys:
            raise ValueError("No SenseNova lora_down tensors were found in the LoRA file.")

        targets = {}
        for down_key in down_keys:
            name = down_key[: -len(_DOWN_SUFFIX)]
            up_key = name + _UP_SUFFIX
            alpha_key = name + _ALPHA_SUFFIX
            if up_key not in keys or alpha_key not in keys:
                raise ValueError(f"SenseNova LoRA target {name} is missing its lora_up or alpha tensor.")
            down = checkpoint.get_tensor(down_key).contiguous()
            up = checkpoint.get_tensor(up_key).contiguous()
            rank = down.shape[0]
            alpha = checkpoint.get_tensor(alpha_key).item()
            targets[name] = LoRATarget(down=down, up=up, scale=alpha / rank)

    return SenseNovaLoRA(path, metadata, targets, strength)


__all__ = ["SenseNovaLoRA", "load_sensenova_lora"]
