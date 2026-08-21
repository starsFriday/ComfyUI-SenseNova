"""Generic FP8 (scaled) dequant + load flow for `quant_fp8_scaled.py` checkpoints.

A model-agnostic ("wildcard") loader that takes ANY nn.Module plus a state dict
produced by ``quant_fp8_scaled.py`` (the legacy "scaled_fp8" layout) and makes the
model load and run those FP8 weights OUTSIDE of ComfyUI's own ops -- i.e. in a plain
transformer / diffusers / HuggingFace flow.

WHY THIS EXISTS
---------------
``quant_fp8_scaled.py`` stores, per 2D weight under the diffusion-model prefix::

    <prefix><layer>.weight       -> torch.float8_e4m3fn  (per OUTPUT-CHANNEL quantized)
    <prefix><layer>.scale_weight -> float32 scale [N] or scalar  (only for scaled modes;
                                      ABSENT when quantized with --scale-mode none)
    <prefix>scaled_fp8           -> scalar marker (1.0 -> full_precision_mm=False)

Scale is OPTIONAL: the quantizer supports ``--scale-mode {per-channel,per-tensor,none}``.
This loader detects scale presence per layer (via the ``.scale_weight`` sibling) and
de-quantizes accordingly: with scale -> ``w = q * scale``; without -> ``w = q.float()``.

ComfyUI auto-detects this on load via ``comfy/utils.py:convert_old_quants`` +
``comfy/quant_ops.py:QuantizedTensor`` (which runs the FP8 tensor-core ``scaled_mm``
path). A *plain* nn.Module has none of that machinery, so loading the raw file would
put ``float8_e4m3fn`` into ``nn.Linear.weight`` (unsupported in eager mode) and leave
``.scale_weight`` unused -> garbage output. This loader reproduces ComfyUI's
de-quantization in a self-contained, framework-agnostic way.

It mirrors the proven 4-stage flow of ``convrot_loader.py`` (detect -> patch modules
-> remap keys -> load_state_dict) and reuses its ``_ConvRotBase`` plumbing, so FP8
layers get the same buffer-dtype protection, optional LoRA merge, and meta-device
streaming support as the ConvRot / GGUF layers.

Typical use (no source edits to the target model required)::

    from fp8_scaled_loader import load_fp8_scaled_state_dict
    model = MyModel(...)                              # fresh module with plain nn.Linear
    info = load_fp8_scaled_state_dict(model, "model_fp8_scaled.safetensors")
    print(info)                                      # {'fp8': N, 'replaced': [...], ...}
    model = model.to(torch.bfloat16).to(device).eval()

Dequant math (identical to the converter's round-trip, verified byte-compatible)::

    scaled modes :  q = w / scale   (quant)   ->   w_deq = q.float() * scale.reshape(-1, 1)
    --scale-mode none :  q = clamp(w)         ->   w_deq = q.float()   (no scale)

The de-quantized weight is produced ONCE per forward (in float compute dtype) and used
immediately, then released -- the tiny fp8 + scale buffers stay resident, so VRAM stays
low (same streaming behavior as ConvRot/GGUF). Only ``float8_e4m3fn`` + ``float32``
buffers are kept; no full bf16 copy is ever persisted.

Optional fast path: set ``use_scaled_mm=True`` to run the matmul through
``torch._scaled_mm`` (FP8 tensor-core) on Ada/Blackwell + torch>=2.4 + CUDA, exactly
like ComfyUI's native path. Falls back to de-quant automatically when unavailable.
"""
from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy_kitchen
from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

# --- reuse the shared plumbing from convrot_loader (buffer protection, LoRA merge,
#     module-tree traversal, meta materialize). convrot_loader has no hard deps that
#     can fail (comfy_kitchen is optional), so this import is safe standalone. ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from .convrot_loader import (  # noqa: E402
    _ConvRotBase,
    _get_parent_and_child,
    load_any_state_dict,
    _materialize_leftover_meta,
    _merge_lora_into_bases,
)


# ===========================================================================
# Self-dequantizing FP8 (scaled) Linear module
# ===========================================================================
class FP8ScaledLinear(_ConvRotBase):
    """FP8 linear, format ``fp8_scaled`` (scale optional).

    Stores the fp8 weight [N, K]. When the checkpoint was quantized with a scaled
    mode (``--scale-mode per-channel|per-tensor``) it also keeps the float32
    ``weight_scale`` buffer; for ``--scale-mode none`` there is no scale buffer and
    de-quantization is just ``w = q.float()``.

    On forward it de-quantizes to the compute dtype and runs a normal ``F.linear``,
    then frees the transient weight. This is the framework-agnostic equivalent of
    ComfyUI's ``QuantizedTensor`` for the scaled_fp8 layout, and keeps the same
    low-VRAM streaming behavior as the ConvRot/GGUF layers.
    """

    # keep fp8 weight (+ fp32 scale when present) exactly as stored
    # (never let .to()/half() cast them)
    _PROTECTED = ("weight_fp8", "weight_scale")

    def __init__(self, in_features, out_features, bias=True, groupsize=256,
                 full_precision_mm: bool = False, use_scaled_mm: bool = False,
                 has_scale: bool = True, scale_shape=None):
        super().__init__(in_features, out_features, bias, groupsize)
        self.register_buffer("weight_fp8", torch.empty((out_features, in_features),
                                                         dtype=torch.float8_e4m3fn))
        self.has_scale = bool(has_scale)
        if self.has_scale:
            # buffer shape follows the stored scale (per-channel [N], per-tensor scalar/[1])
            sshape = tuple(scale_shape) if scale_shape is not None else (out_features,)
            self.register_buffer("weight_scale", torch.empty(sshape, dtype=torch.float32))
        else:
            self.weight_scale = None  # not a persistent buffer; dequant = q.float()
        self.full_precision_mm = bool(full_precision_mm)
        self.use_scaled_mm = bool(use_scaled_mm)

    @torch.no_grad()
    def _get_original_weight(self) -> torch.Tensor:
        if self.has_scale:
            # dequant: q = w / scale  ->  w = q * scale  (per output channel / per tensor)
            return self.weight_fp8.float() * self.weight_scale.reshape(-1, 1)
        # --scale-mode none: plain fp8, no scale
        return self.weight_fp8.float()

    def forward(self, x):
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None

        capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else (0, 0)
        if (self.use_scaled_mm and not self.full_precision_mm and self.has_scale
                and self.weight_scale.numel() == 1 and capability >= (8, 9)
                and not (self._has_lora_buffer and self._lora_enabled)):
            input_shape = x.shape
            x = x.reshape(-1, input_shape[-1])
            input_scale = (x.abs().amax().float() / 448.0).clamp(min=1e-12)
            x_fp8 = comfy_kitchen.quantize_per_tensor_fp8(x, input_scale)
            out = scaled_mm_v2(
                x_fp8,
                self.weight_fp8.to(x.device).T,
                input_scale,
                self.weight_scale.to(x.device),
                bias=b,
                out_dtype=x.dtype,
            )
            return out.reshape(*input_shape[:-1], self.out_features)

        w = self._get_original_weight()                      # original-space weight (float32)
        w = w.to(dtype=x.dtype, device=x.device)             # de-quant into compute space
        if self._has_lora_buffer and self._lora_enabled:
            delta = (self._lora_B_buf @ self._lora_A_buf) * self._lora_scaling
            w = (w + delta.to(device=w.device)).to(dtype=w.dtype)

        out = F.linear(x, w, b)
        del w  # release the transient de-quantized weight immediately
        return out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, fp8_scaled=True, "
                f"has_scale={self.has_scale}, "
                f"full_precision_mm={self.full_precision_mm}, "
                f"use_scaled_mm={self.use_scaled_mm}")


# format string -> (module class, remapped weight-buffer key)
_FORMAT_TABLE = {
    "fp8_scaled": (FP8ScaledLinear, "weight_fp8"),
}


# ===========================================================================
# Detection
# ===========================================================================
def detect_fp8_scaled_layers(state_dict: dict) -> dict:
    """Scan a state dict for FP8 (scaled_fp8) quantized layers.

    Detection is driven by the ``scaled_fp8`` marker (written by quant_fp8_scaled.py):
    its presence scopes the prefix, and every ``<base>.weight`` that is
    ``float8_e4m3fn`` is a quantized layer. Scale is OPTIONAL: a layer quantized with
    ``--scale-mode none`` has NO ``.scale_weight`` sibling and de-quantizes as
    ``w = q.float()``; scaled layers carry ``.scale_weight`` and de-quantize as
    ``w = q * scale``. ``has_scale`` records this per layer.

    Returns ``{base: {"format": "fp8_scaled", "full_precision_mm": bool,
    "has_scale": bool}}``. Works on any checkpoint regardless of the model it came from.
    """
    # 1) locate the marker(s) and derive the prefix(es)
    prefixes = []
    full_precision_mm = False
    for k, v in state_dict.items():
        if k.endswith("scaled_fp8"):
            prefixes.append(k[: -len("scaled_fp8")])
            try:
                if v.nelement() == 2:
                    full_precision_mm = True
            except Exception:
                pass
    marker_scoped = bool(prefixes)
    if not prefixes:
        # fallback: any float8 weight with a sibling .scale_weight counts
        for k, v in state_dict.items():
            if k.endswith(".weight") and getattr(v, "dtype", None) == torch.float8_e4m3fn:
                base = k[: -len(".weight")]
                if f"{base}.scale_weight" in state_dict:
                    prefixes.append("")  # unscoped
    if not prefixes:
        return {}

    plan = {}
    seen = set()
    for prefix in prefixes:
        for k, v in state_dict.items():
            if not k.startswith(prefix) or not k.endswith(".weight"):
                continue
            base = k[: -len(".weight")]
            if base in seen:
                continue
            if getattr(v, "dtype", None) != torch.float8_e4m3fn:
                continue
            has_scale = f"{base}.scale_weight" in state_dict
            # fallback (unscoped, no marker) only trusts fp8 weights that carry a scale,
            # to avoid mistaking unrelated fp8 tensors for our format.
            if not marker_scoped and not has_scale:
                continue
            seen.add(base)
            entry = {"format": "fp8_scaled",
                     "full_precision_mm": full_precision_mm,
                     "has_scale": has_scale}
            if has_scale:
                entry["scale_shape"] = tuple(state_dict[f"{base}.scale_weight"].shape)
            plan[base] = entry
    return plan


# ===========================================================================
# Module-tree surgery (reuses convrot_loader's dotted-path traversal)
# ===========================================================================
def patch_model_with_fp8(model: nn.Module, plan: dict, verbose: bool = True,
                         on_meta: bool = False, use_scaled_mm: bool = False) -> list:
    """Replace each planned nn.Linear in ``model`` with an ``FP8ScaledLinear``.

    Must run BEFORE ``model.to(dtype)`` so the fp8/scale buffers keep their dtype (the
    module overrides ``_apply`` to protect them). Returns the list of replaced base paths.
    Safe/idempotent: already-converted or missing layers are skipped.

    ``on_meta=True`` builds the replacement layers on the meta device (zero real
    allocation) -- used by :func:`load_fp8_scaled_into_meta_model`.
    """
    replaced = []
    ctx = torch.device("meta") if on_meta else contextlib.nullcontext()
    for base, meta in plan.items():
        parent, child = _get_parent_and_child(model, base)
        if parent is None:
            continue
        module = getattr(parent, child, None)
        if module is None or isinstance(module, _ConvRotBase):
            continue
        if not isinstance(module, nn.Linear):
            continue
        cls, _ = _FORMAT_TABLE[meta["format"]]
        with ctx:
            new_layer = cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                full_precision_mm=meta.get("full_precision_mm", False),
                use_scaled_mm=use_scaled_mm,
                has_scale=meta.get("has_scale", True),
                scale_shape=meta.get("scale_shape"),
            )
        setattr(parent, child, new_layer)
        replaced.append(base)
    if verbose:
        print(f"[FP8] replaced {len(replaced)} Linear layer(s) with FP8ScaledLinear",
              file=sys.stderr)
    return replaced


def remap_fp8_state_dict_keys(state_dict: dict, plan: dict, drop_marker: bool = True) -> dict:
    """Rename fp8-scaled keys for the module buffers.

    - ``{base}.weight``      -> ``{base}.weight_fp8``
    - ``{base}.scale_weight`` -> ``{base}.weight_scale``  (kept as-is, just renamed)
    ``{prefix}scaled_fp8`` marker is dropped. Mutates and returns the same dict.
    """
    for base in plan:
        w_key = f"{base}.weight"
        if w_key in state_dict:
            state_dict[f"{base}.weight_fp8"] = state_dict.pop(w_key)
        sw_key = f"{base}.scale_weight"
        if sw_key in state_dict:
            state_dict[f"{base}.weight_scale"] = state_dict.pop(sw_key)
        if drop_marker:
            for mk in [k for k in state_dict if k.endswith("scaled_fp8")]:
                state_dict.pop(mk)
    return state_dict


# ===========================================================================
# High-level one-shot API
# ===========================================================================
def load_fp8_scaled_state_dict(
    model: nn.Module,
    checkpoint,
    *,
    device: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
    use_scaled_mm: bool = False,
) -> dict:
    """One-shot: detect -> patch modules -> remap keys -> load_state_dict.

    ``checkpoint`` may be a path (str) or an already-loaded state dict.
    Returns an info dict: counts, replaced bases, and load_state_dict's
    (missing_keys, unexpected_keys). Non-quantized checkpoints load normally.

    IMPORTANT: call this BEFORE moving the model to its compute dtype, e.g.::

        info = load_fp8_scaled_state_dict(model, path)
        model = model.to(torch.bfloat16).to(device).eval()

    The FP8ScaledLinear modules protect their fp8/scale buffers from the later ``.to()``.
    """
    state_dict = load_any_state_dict(checkpoint, device=device) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_fp8_scaled_layers(state_dict)
    if verbose:
        print(f"[FP8] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_fp8(model, plan, verbose=verbose,
                                    use_scaled_mm=use_scaled_mm) if plan else []
    active_plan = {b: plan[b] for b in replaced}
    remap_fp8_state_dict_keys(state_dict, active_plan, drop_marker=True)

    result = model.load_state_dict(state_dict, strict=strict)
    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected):
        print(f"[FP8] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", file=sys.stderr)

    return {
        "fp8": len(replaced),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def load_fp8_scaled_into_meta_model(
    model: nn.Module,
    checkpoint,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = False,
    verbose: bool = True,
    use_scaled_mm: bool = False,
) -> dict:
    """Memory-lean FP8 load for a model built on the **meta** device.

    Build the model on meta (zero real allocation), then let the compact fp8 bytes (plus
    the small non-quant tensors) be *assigned* straight into it with
    ``load_state_dict(assign=True)``. Only the tiny fp8 + scale buffers ever hit RAM; the
    full bf16 weight of each layer is produced lazily on its first forward, never all at
    once at load time. Mirrors :func:`load_convrot_into_meta_model`::

        with torch.device("meta"):
            model = MyModel(config)                  # zero real memory
        info = load_fp8_scaled_into_meta_model(model, "model_fp8_scaled.safetensors",
                                               device="cpu", dtype=torch.bfloat16)
        model = model.to(device).eval()              # already on `device`; ready to run

    The fp8 buffer keeps its float8 dtype and ``weight_scale`` stays float32 (not
    downcast -> de-quant precision); plain fp weights are cast to ``dtype`` on ``device``.
    Any keys the checkpoint doesn't cover are materialized as uninitialized tensors and
    reported under ``meta_materialized``. Requires torch>=2.1 (load_state_dict assign).
    """
    state_dict = load_any_state_dict(checkpoint, device=device) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_fp8_scaled_layers(state_dict)
    if verbose:
        print(f"[FP8-meta] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_fp8(model, plan, verbose=verbose, on_meta=True,
                                    use_scaled_mm=use_scaled_mm) if plan else []
    active_plan = {b: plan[b] for b in replaced}
    remap_fp8_state_dict_keys(state_dict, active_plan, drop_marker=True)

    # Prepare tensors for assign: fp8 stays fp8; scale stays fp32; plain fp -> dtype.
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".weight_fp8"):
            state_dict[k] = v.to(device=device).contiguous()              # keep float8
        elif k.endswith(".weight_scale"):
            state_dict[k] = v.to(device=device, dtype=torch.float32).contiguous()  # keep fp32
        elif v.is_floating_point():
            state_dict[k] = v.to(device=device, dtype=dtype)
        else:
            state_dict[k] = v.to(device=device)

    result = model.load_state_dict(state_dict, strict=strict, assign=True)
    leftover = _materialize_leftover_meta(model, device, dtype)

    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected or leftover):
        print(f"[FP8-meta] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected, {len(leftover)} meta-materialized", file=sys.stderr)

    return {
        "fp8": len(replaced),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "meta_materialized": leftover,
    }


def load_fp8_scaled_lora(model: nn.Module, lora_state_dict: dict, alpha=None, r=None):
    """Merge LoRA into FP8ScaledLinear layers (de-quant-time merge).

    Mirrors :func:`load_gguf_lora` / :func:`load_convrot_lora`: the delta is stored on
    the module (``_lora_A_buf`` / ``_lora_B_buf``) and merged into the de-quantized
    weight on every forward (see ``FP8ScaledLinear.forward``). Both the standard
    ``lora_down``/``lora_up`` layout and the A/B layout are supported.

    Plain ``nn.Linear`` layers (e.g. the bf16-kept embed_tokens / lm_head / heads) are
    NOT touched here -- call the model's own ``load_and_merge_lora_weight_from_safetensors``
    for those. Returns ``(merged_bases, leftover_bases)``.
    """
    return _merge_lora_into_bases(model, lora_state_dict, alpha, r, tag="FP8")


__all__ = [
    "FP8ScaledLinear",
    "detect_fp8_scaled_layers",
    "patch_model_with_fp8",
    "remap_fp8_state_dict_keys",
    "load_fp8_scaled_state_dict",
    "load_fp8_scaled_into_meta_model",
    "load_fp8_scaled_lora",
]
