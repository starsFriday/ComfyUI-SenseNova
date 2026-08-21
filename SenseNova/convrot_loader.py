"""Generic ConvRot dequant + load flow for comfy-kitchen quantized checkpoints.

A model-agnostic ("wildcard") loader that takes ANY nn.Module plus a state dict
produced by ``quant_int8_convrot.py`` (format ``int8_tensorwise``) or
``quant_int4_convrot.py`` (format ``convrot_w4a4``) and makes the model load and
run those quantized weights outside of ComfyUI's own ops.

It borrows the proven pattern used inside VoxCPM2.from_local (detect ``.comfy_quant``
markers -> swap the matching nn.Linear for a self-dequantizing Linear -> remap the
state-dict keys), but generalized so it works on any architecture and BOTH bit widths,
WITHOUT importing from or modifying voxcpm2.py.

Typical use (no source edits to the target model required)::

    from convrot_loader import load_convrot_state_dict
    model = MyModel(...)                       # fresh module with plain nn.Linear
    info = load_convrot_state_dict(model, "model_int4_convrot.safetensors")
    print(info)                               # {'int8': N, 'int4': M, ...}
    model = model.to(device).eval()

The de-quantized weight is computed once (Hadamard un-rotation) and cached in a bf16
buffer (mirrors VoxCPM2.ConvRotLinear.weight_cached), so each forward only pays a cheap
F.linear. Optional LoRA can be merged into convrot layers at de-quant time via
``load_convrot_lora(model, lora_state_dict)`` (keys ``<path>.lora_A`` / ``<path>.lora_B``).

Or step by step::

    sd = load_any_state_dict(path)
    plan = detect_convrot_layers(sd)          # {base: {format, groupsize}}
    patch_model_with_convrot(model, plan)     # replace nn.Linear -> ConvRot*Linear
    sd = remap_state_dict_keys(sd, plan)      # .weight -> .weight_int8/.weight_packed
    model.load_state_dict(sd, strict=False)

Dequant math is identical to the converters' own round-trip (verified byte-compatible):
  int8:  w_rot = int8 * scale;                 w = un_rotate(w_rot, H, gs)
  int4:  w_rot = unpack_int4(packed) * scale;  w = un_rotate(w_rot, H, gs)
The block-Hadamard rotation is an involution for the normalized Hadamard, so applying
``_rotate_weight`` again maps rotated space back to the original space.
"""
from __future__ import annotations

import contextlib
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy_kitchen

# --- rotation primitives (same ones both converters use) ---
try:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight
except ImportError:  # pragma: no cover
    try:
        from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
    except ImportError:  # pragma: no cover - self-contained mirror of int8_utils
        def _build_hadamard(
            size: int,
            device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

            Fallback mirror of comfy_kitchen.tensor.int8_utils._build_hadamard so
            the loader works even when comfy_kitchen is not importable.
            """
            if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
                raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
            h4 = torch.tensor(
                [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                dtype=dtype,
                device=device,
            )
            h = h4
            current_size = 4
            while current_size < size:
                h = torch.kron(h, h4)
                current_size *= 4
            return h / (size**0.5)

        def _rotate_weight(
            weight: torch.Tensor,
            h: torch.Tensor,
            group_size: int,
        ) -> torch.Tensor:
            """Rotate weight matrix offline: W_rot = W @ H_block^T.

            Fallback mirror of comfy_kitchen.tensor.int8_utils._rotate_weight.
            """
            out_f, in_f = weight.shape
            if in_f % group_size != 0:
                raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
            n_groups = in_f // group_size
            weight_grouped = weight.reshape(out_f, n_groups, group_size)
            h_t = h.T.to(dtype=weight.dtype, device=weight.device)
            weight_rotated = torch.matmul(weight_grouped, h_t)
            return weight_rotated.reshape(out_f, in_f)

# --- int4 pack/unpack codec (with a self-contained fallback mirror) ---
try:
    from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
except ImportError:  # pragma: no cover - fallback mirror of the int4 codec
    def _unpack_int4_row_major(packed):
        x32 = packed.to(torch.int32)
        lo = x32 & 0x0F
        hi = (x32 >> 4) & 0x0F
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)

# --- GGUF dequant (self-contained, NO diffusers dependency) ------------
# The dequant kernels below are the pure-torch ports from City96 / ComfyUI-GGUF
# (see ComfyUI_Dif_GGUF/ggml_ops.py, Apache-2.0). We inline them so this loader only
# needs the lightweight `gguf` package (for GGML_QUANT_SIZES + the quantization-type
# enum) -- not the heavy `diffusers` install. This section is fully guarded so the
# ConvRot path keeps working even when `gguf` is absent.
_GGUF_IMPORT_ERROR = None
try:
    import gguf as _gguf
except Exception as _e:  # pragma: no cover - gguf missing
    _gguf = None
    _GGUF_IMPORT_ERROR = _e

_GGUF_UNQUANT = set()
if _gguf is not None:
    _GGUF_UNQUANT = {
        _gguf.GGMLQuantizationType.F32,
        _gguf.GGMLQuantizationType.F16,
        _gguf.GGMLQuantizationType.BF16,
    }


def _require_gguf():
    if _gguf is None:
        raise ImportError(
            "GGUF support requires the `gguf` package (pip install gguf). "
            f"Original import error: {_GGUF_IMPORT_ERROR}"
        )


def _to_qtype(quant_type):
    """Normalize an int / enum quant type to a gguf.GGMLQuantizationType member."""
    _require_gguf()
    return _gguf.GGMLQuantizationType(int(quant_type))


# ── Dequant kernels (torch-only, City96 / ComfyUI-GGUF) ─────────────────
QK_K = 256
K_SCALE_SIZE = 12


def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs


def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor([2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)


dequantize_functions = {}
if _gguf is not None:
    dequantize_functions = {
        _gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
        _gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
        _gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
        _gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
        _gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
        _gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
        _gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
        _gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
        _gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
        _gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
        _gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
        _gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
        _gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    }

_GGML_QUANT_SIZES = _gguf.GGML_QUANT_SIZES if _gguf is not None else {}
_GGUF_DEQUANT_FNS = dequantize_functions


class _GGUFParameter(torch.Tensor):
    """Minimal stand-in for diffusers' GGUFParameter.

    A tensor subclass that carries its ``quant_type`` so ``detect_gguf_layers`` can
    spot quantized weights, without pulling in diffusers. ``as_tensor`` flattens it
    back to a plain tensor for ``load_state_dict`` / dequant.
    """

    def __new__(cls, data, quant_type):
        t = torch.as_tensor(data) if not isinstance(data, torch.Tensor) else data
        t = torch.Tensor._make_subclass(cls, t, False)
        t.quant_type = quant_type
        return t

    def as_tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self, False)


@torch.no_grad()
def _dequant_gguf_bytes(raw_uint8: torch.Tensor, quant_type) -> torch.Tensor:
    """De-quantize raw GGUF weight bytes -> float tensor of logical shape [out, in].

    Self-contained equivalent of diffusers' dequantize_gguf_tensor, operating on a
    plain uint8 buffer + an explicit quant_type (no live GGUFParameter wrapper needed).
    """
    _require_gguf()
    qt = _to_qtype(quant_type)
    block_size, type_size = _GGML_QUANT_SIZES[qt]
    t = raw_uint8.view(torch.uint8)
    shape = (*t.shape[:-1], t.shape[-1] // type_size * block_size)
    n_blocks = t.numel() // type_size
    blocks = t.reshape((n_blocks, type_size))
    deq = _GGUF_DEQUANT_FNS[qt](blocks, block_size, type_size)
    return deq.reshape(shape)


# Hadamard groupsizes to fall back through if the stored gs doesn't divide K.
_GS_FALLBACK = (256, 64, 16)


def _resolve_gs(k: int, preferred: int) -> int:
    """Pick a Hadamard groupsize that divides K, preferring the stored value."""
    if preferred and k % preferred == 0:
        return preferred
    for g in _GS_FALLBACK:
        if k % g == 0:
            return g
    return preferred or _GS_FALLBACK[-1]


# ===========================================================================
# Self-dequantizing Linear modules (one per bit width)
# ===========================================================================
class _ConvRotBase(nn.Module):
    """Shared plumbing: buffer-dtype protection + (cached) un-rotate + F.linear forward.

    The de-quantized weight is computed once (Hadamard un-rotation) and cached in the
    de-quantized on the fly ("use and discard") -- the heavy bf16 weight is produced
    in ``forward`` and released right after the matmul, so no full bf16 copy is ever
    kept resident. Optional LoRA is merged into that weight at forward time via
    ``load_lora_buffer`` (interface parity with VoxCPM2).
    """

    # names of buffers that must NEVER be dtype-cast by .to()/.half()/.bfloat16()
    _PROTECTED: tuple = ()

    def __init__(self, in_features, out_features, bias, groupsize):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = int(groupsize) if groupsize else 256
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        # NOTE: we deliberately keep NO persistent bf16 copy of the de-quantized
        # weight. The heavy (~model-sized) bf16 weight is produced on the fly in
        # ``forward`` (see "use and discard" below) and released right after the
        # matmul, so the full de-quantized model never has to reside on the GPU at
        # once. Only the tiny quantized buffers (weight_gguf / weight_int8 /
        # weight_packed) stay resident, which is what makes low-VRAM / layer-
        # streaming inference work. The old ``weight_cached`` cache is removed.
        # optional LoRA merged at de-quant time (hook; see load_lora_buffer)
        self._has_lora_buffer = False
        self._lora_enabled = True

    def _apply(self, fn, recurse=True):
        """Protect integer/scale buffers from dtype conversion (keep device sync)."""
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is None:
                continue
            if key in self._PROTECTED:
                self._buffers[key] = buf.to(device=fn(buf).device)  # move only, no cast
            else:
                self._buffers[key] = fn(buf)
        return self

    @torch.no_grad()
    def _get_original_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def mark_dequant_dirty(self):
        """No-op kept for interface parity.

        The de-quantized weight is no longer cached (see ``forward``), so there is
        nothing to invalidate. Kept so callers that edited quantized buffers in place
        do not break.
        """
        return

    # ---- optional LoRA merge-at-dequant (mirrors VoxCPM2.ConvRotLinear) ----
    def load_lora_buffer(self, lora_A, lora_B, alpha, r):
        """Store LoRA (A:[r,in], B:[out,r]) and merge it into the de-quantized weight on
        every forward. The base de-quant cache stays valid; only the cheap LoRA delta is
        recomputed at forward time.
        """
        # weight_cached may be None before the first forward (meta-load path), so
        # derive the LoRA device from a quantized buffer that is always present.
        dev = None
        for buf_name in self._PROTECTED:
            buf = getattr(self, buf_name, None)
            if buf is not None:
                dev = buf.device
                break
        if dev is None:
            dev = self.bias.device if self.bias is not None else torch.device("cpu")
        # LoRA tensors are float32 in the checkpoint; keep them float32 so the merged
        # delta is accurate (a bf16 cast would quantize away the small LoRA deltas).
        self.register_buffer("_lora_A_buf", lora_A.contiguous().to(device=dev, dtype=torch.float32))
        self.register_buffer("_lora_B_buf", lora_B.contiguous().to(device=dev, dtype=torch.float32))
        self._lora_scaling = alpha / r
        self._has_lora_buffer = True

    def set_lora_buffer_enabled(self, enabled: bool):
        """Enable/disable the LoRA delta (interface parity with LoRALinear)."""
        self._lora_enabled = enabled

    def forward(self, x):
        # "Use and discard" streaming-friendly forward: de-quantize the original-space
        # weight on the fly, run the matmul, then free it. We never persist a full bf16
        # copy, so the model's de-quantized weight only ever exists for a single layer's
        # forward (the tiny quantized buffers stay resident). This is the canonical GGUF
        # inference behavior and is what makes layer-streaming / low-VRAM work.
        w = self._get_original_weight()                  # original-space weight (float)
        w = w.to(dtype=x.dtype, device=x.device)         # de-quant into compute space
        if self._has_lora_buffer and self._lora_enabled:
            # LoRA delta is computed in float32 (the LoRA tensors are float32 in the
            # checkpoint) and upcast onto the bf16 base for an accurate merge.
            delta = (self._lora_B_buf @ self._lora_A_buf) * self._lora_scaling
            w = (w + delta.to(device=w.device)).to(dtype=w.dtype)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        out = F.linear(x, w, b)
        del w  # release the transient de-quantized weight immediately
        return out


class ConvRotInt8Linear(_ConvRotBase):
    """INT8 + ConvRot linear (format ``int8_tensorwise``).

    Stores int8 weight [N, K] and per-output-channel scale [N, 1].
    """

    _PROTECTED = ("weight_int8", "weight_scale")

    def __init__(self, in_features, out_features, bias=True, groupsize=256):
        super().__init__(in_features, out_features, bias, groupsize)
        self.register_buffer("weight_int8", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

    @torch.no_grad()
    def _get_original_weight(self):
        w_rot = self.weight_int8.float() * self.weight_scale.reshape(-1, 1)  # rotated space
        gs = _resolve_gs(w_rot.shape[1], self.convrot_groupsize)
        h = _build_hadamard(gs, device=w_rot.device, dtype=torch.float32)
        return _rotate_weight(w_rot, h, gs)  # -> original space

    def forward(self, x):
        if self._has_lora_buffer and self._lora_enabled:
            return super().forward(x)
        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return comfy_kitchen.int8_linear(
            x,
            self.weight_int8,
            self.weight_scale,
            bias,
            x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, convrot_gs={self.convrot_groupsize}, int8=True")


class ConvRotInt4Linear(_ConvRotBase):
    """INT4 + ConvRot linear (format ``convrot_w4a4``).

    Stores packed int4 weight as int8 [N, K//2] (2 nibbles/byte) and
    per-output-channel scale [N] (1-D, matches the converter output).
    """

    _PROTECTED = ("weight_packed", "weight_scale")

    def __init__(self, in_features, out_features, bias=True, groupsize=256):
        super().__init__(in_features, out_features, bias, groupsize)
        self.register_buffer("weight_packed", torch.empty((out_features, in_features // 2), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features,), dtype=torch.float32))

    @torch.no_grad()
    def _get_original_weight(self):
        qint = _unpack_int4_row_major(self.weight_packed)                 # signed int [N, K]
        w_rot = qint.float() * self.weight_scale.reshape(-1, 1)           # rotated space
        gs = _resolve_gs(w_rot.shape[1], self.convrot_groupsize)
        h = _build_hadamard(gs, device=w_rot.device, dtype=torch.float32)
        return _rotate_weight(w_rot, h, gs)                              # -> original space

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, convrot_gs={self.convrot_groupsize}, int4=True")


class GGUFDequantLinear(_ConvRotBase):
    """GGUF (GGML) quantized linear (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/...).

    Stores the raw GGUF weight bytes as a uint8 buffer [out, byte_cols] plus the integer
    ``quant_type``. On every forward the bytes are de-quantized (torch kernels ported
    from City96, shipped by diffusers) into the [out, in] weight on the fly and used
    immediately, then discarded -- no persistent bf16 copy is kept (streaming-friendly).
    Optional LoRA merges at forward time exactly like the ConvRot layers.
    """

    _PROTECTED = ("weight_gguf",)

    def __init__(self, in_features, out_features, quant_type, byte_cols, bias=True, groupsize=256):
        super().__init__(in_features, out_features, bias, groupsize)
        self.quant_type = int(quant_type)
        self.register_buffer("weight_gguf", torch.empty((out_features, int(byte_cols)), dtype=torch.uint8))

    @torch.no_grad()
    def _get_original_weight(self):
        w = _dequant_gguf_bytes(self.weight_gguf, self.quant_type)   # [out, in], float
        return w.float()

    def extra_repr(self):
        try:
            qname = _to_qtype(self.quant_type).name
        except Exception:
            qname = str(self.quant_type)
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, gguf={qname}")


# format string -> (module class, remapped weight-buffer key)
_FORMAT_TABLE = {
    "int8_tensorwise": (ConvRotInt8Linear, "weight_int8"),
    "convrot_w4a4":    (ConvRotInt4Linear, "weight_packed"),
}


# ===========================================================================
# Detection
# ===========================================================================
def _parse_comfy_quant(tensor) -> dict | None:
    """Decode a ``.comfy_quant`` uint8 tensor back into its JSON config dict."""
    try:
        raw = bytes(tensor.cpu().numpy().tobytes())
        return json.loads(raw)
    except Exception:
        return None


def detect_convrot_layers(state_dict: dict) -> dict:
    """Scan a state dict for ConvRot-quantized layers.

    Returns ``{base: {"format": str, "groupsize": int, "convrot": bool}}`` for every
    ``.comfy_quant`` marker whose format is supported (int8_tensorwise / convrot_w4a4).
    Works on any checkpoint regardless of the model it came from.
    """
    plan = {}
    for k in list(state_dict.keys()):
        if not k.endswith(".comfy_quant"):
            continue
        base = k[: -len(".comfy_quant")]
        conf = _parse_comfy_quant(state_dict[k])
        if not conf:
            continue
        fmt = conf.get("format")
        if fmt not in _FORMAT_TABLE:
            continue
        plan[base] = {
            "format": fmt,
            "groupsize": int(conf.get("convrot_groupsize", 256)),
            "convrot": bool(conf.get("convrot", True)),
        }
    return plan


# ===========================================================================
# Module-tree surgery (dotted-path traversal, handles nn.ModuleList indices)
# ===========================================================================
def _get_parent_and_child(model: nn.Module, base: str):
    """Return (parent_module, child_attr_name) for a dotted path, or (None, None)."""
    parts = base.split(".")
    parent = model
    for p in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            try:
                parent = parent[int(p)]
            except (ValueError, IndexError):
                return None, None
        else:
            parent = getattr(parent, p, None)
        if parent is None:
            return None, None
    return parent, parts[-1]


def patch_model_with_convrot(model: nn.Module, plan: dict, verbose: bool = True,
                              on_meta: bool = False) -> list:
    """Replace each planned nn.Linear in ``model`` with the matching ConvRot*Linear.

    Must run BEFORE ``model.to(dtype)`` so the int8/int4 buffers keep their integer
    dtype (the modules override ``_apply`` to protect them). Returns the list of
    successfully replaced base paths. Safe/idempotent: already-converted or missing
    layers are skipped.

    ``on_meta=True`` builds the replacement layers on the meta device (zero real
    allocation) so a meta-loaded model stays meta until the checkpoint is assigned into
    it -- used by :func:`load_convrot_into_meta_model`.
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
                groupsize=meta["groupsize"],
            )
        setattr(parent, child, new_layer)
        replaced.append(base)
    if verbose:
        n8 = sum(1 for b in replaced if plan[b]["format"] == "int8_tensorwise")
        n4 = sum(1 for b in replaced if plan[b]["format"] == "convrot_w4a4")
        print(f"[ConvRot] replaced {len(replaced)} Linear layers "
              f"(int8={n8}, int4={n4})", file=sys.stderr)
    return replaced


def remap_state_dict_keys(state_dict: dict, plan: dict, drop_marker: bool = True) -> dict:
    """Rename ``{base}.weight`` -> the module's weight buffer and drop markers.

    - int8: ``{base}.weight`` -> ``{base}.weight_int8``
    - int4: ``{base}.weight`` -> ``{base}.weight_packed``
    ``{base}.weight_scale`` is kept as-is. Mutates and returns the same dict.
    """
    for base, meta in plan.items():
        _, buf_key = _FORMAT_TABLE[meta["format"]]
        w_key = f"{base}.weight"
        if w_key in state_dict:
            state_dict[f"{base}.{buf_key}"] = state_dict.pop(w_key)
        if drop_marker:
            state_dict.pop(f"{base}.comfy_quant", None)
    return state_dict


# ===========================================================================
# GGUF: detection / module-surgery / key-remap (same 4-stage flow as ConvRot)
# ===========================================================================
def detect_gguf_layers(state_dict: dict) -> dict:
    """Scan a GGUF state dict for quantized linear weights.

    A layer counts as GGUF-quantized when ``{base}.weight`` is a ``GGUFParameter`` (carries
    ``.quant_type``) whose type is a real quant type (not F16/F32/BF16). Plain weights load
    normally and are ignored here.

    Returns ``{base: {"quant_type": int, "byte_cols": int, "in_features": int,
    "out_features": int}}``.
    """
    plan = {}
    for k, v in state_dict.items():
        if not k.endswith(".weight"):
            continue
        qt = getattr(v, "quant_type", None)
        if qt is None or qt in _GGUF_UNQUANT:
            continue
        base = k[: -len(".weight")]
        qshape = getattr(v, "quant_shape", None)
        if qshape is not None and len(qshape) == 2:
            out_f, in_f = int(qshape[0]), int(qshape[1])
        else:  # fall back: infer in_features from byte layout
            try:
                block_size, type_size = _GGML_QUANT_SIZES[_to_qtype(qt)]
                out_f = int(v.shape[0])
                in_f = int(v.shape[-1]) // type_size * block_size
            except Exception:
                continue
        plan[base] = {
            "quant_type": int(qt),
            "byte_cols": int(v.shape[-1]),
            "in_features": in_f,
            "out_features": out_f,
        }
    return plan


def patch_model_with_gguf(model: nn.Module, plan: dict, verbose: bool = True,
                          on_meta: bool = False) -> list:
    """Replace each planned nn.Linear in ``model`` with a GGUFDequantLinear.

    Must run BEFORE ``model.to(dtype)`` so the uint8 GGUF buffer keeps its dtype (the module
    overrides ``_apply`` to protect it). Returns the list of replaced base paths.

    ``on_meta=True`` builds the replacement layers on the meta device (zero real
    allocation) so a meta-loaded model stays meta until the checkpoint is assigned into
    it -- used by :func:`load_gguf_into_meta_model`.
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
        with ctx:
            new_layer = GGUFDequantLinear(
                module.in_features,
                module.out_features,
                quant_type=meta["quant_type"],
                byte_cols=meta["byte_cols"],
                bias=module.bias is not None,
            )
        setattr(parent, child, new_layer)
        replaced.append(base)
    if verbose:
        print(f"[GGUF] replaced {len(replaced)} Linear layers"
              f"{' (meta)' if on_meta else ''}", file=sys.stderr)
    return replaced


def remap_gguf_state_dict_keys(state_dict: dict, plan: dict) -> dict:
    """Rename ``{base}.weight`` (GGUFParameter) -> ``{base}.weight_gguf`` (plain uint8).

    Converts the GGUFParameter subclass to a plain uint8 tensor so ``load_state_dict`` copies
    it straight into the module's ``weight_gguf`` buffer. Mutates and returns the dict.
    """
    for base in plan:
        w_key = f"{base}.weight"
        if w_key in state_dict:
            v = state_dict.pop(w_key)
            raw = v.as_tensor() if hasattr(v, "as_tensor") else v
            state_dict[f"{base}.weight_gguf"] = raw.view(torch.uint8).contiguous()
    return state_dict


# ===========================================================================
# I/O helper + high-level one-shot API
# ===========================================================================
def load_any_state_dict(path: str, device: str = "cpu") -> dict:
    """Load a .safetensors / .pt / .pth / .ckpt checkpoint into a flat state dict."""
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=device)
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module", "net", "ema", "params"):
            sub = obj.get(key)
            if isinstance(sub, dict) and any(isinstance(v, torch.Tensor) for v in sub.values()):
                return sub
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(f"no tensor state-dict found in {path}")


def load_gguf_state_dict(path: str) -> dict:
    """Read a .gguf checkpoint into a flat state dict.

    Quantized tensors become ``GGUFParameter`` (carry ``.quant_type``); F16/F32/BF16
    tensors are returned as plain tensors. Mirrors VoxCPM2.load_gguf_checkpoint but
    self-contained. Keys are the raw GGUF tensor names (usually ``<module>.weight`` /
    ``<module>.bias``), matching the model's state-dict paths.
    """
    _require_gguf()
    from gguf import GGUFReader
    bf16_type = _gguf.GGMLQuantizationType.BF16
    reader = GGUFReader(path)
    sd = {}
    for tensor in reader.tensors:
        weights = torch.tensor(tensor.data)
        qt = tensor.tensor_type
        if qt == bf16_type:
            # The gguf reader has no numpy bf16 dtype, so a BF16 tensor comes back as raw
            # uint8 bytes in the *byte* shape [out, in*2] -- already row-major [out, in]
            # like a torch Linear weight (tensor.data uses reversed ggml dims). Just
            # reinterpret each little-endian 2-byte word as bf16; do NOT reshape to
            # tensor.shape, which is the un-reversed ggml `ne` order [in, out] and would
            # transpose every non-square layer (e.g. [3072,4096] -> [4096,3072]).
            sd[tensor.name] = weights.contiguous().view(torch.bfloat16)
        elif qt in _GGUF_UNQUANT:
            sd[tensor.name] = weights
        else:
            sd[tensor.name] = _GGUFParameter(weights, quant_type=qt)
    del reader
    return sd


def load_convrot_state_dict(
    model: nn.Module,
    checkpoint,
    *,
    device: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """One-shot: detect -> patch modules -> remap keys -> load_state_dict.

    ``checkpoint`` may be a path (str) or an already-loaded state dict.
    Returns an info dict: counts, replaced bases, and load_state_dict's
    (missing_keys, unexpected_keys). Non-quantized checkpoints load normally.

    IMPORTANT: call this BEFORE moving the model to its compute dtype, e.g.::

        info = load_convrot_state_dict(model, path)
        model = model.to(torch.bfloat16).to(device).eval()

    The ConvRot modules protect their int8/int4 buffers from the later ``.to()``.
    """
    state_dict = load_any_state_dict(checkpoint, device=device) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_convrot_layers(state_dict)
    if verbose:
        print(f"[ConvRot] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_convrot(model, plan, verbose=verbose) if plan else []
    # only remap the layers we actually replaced (a planned layer may be absent in this model)
    active_plan = {b: plan[b] for b in replaced}
    remap_state_dict_keys(state_dict, active_plan, drop_marker=True)
    # also strip markers for planned-but-not-replaced layers to avoid unexpected keys
    for base in plan:
        if base not in active_plan:
            state_dict.pop(f"{base}.comfy_quant", None)

    result = model.load_state_dict(state_dict, strict=strict)
    # drop the loader's own internal buffers from the "missing" report so callers only
    # see genuinely-unexpected missing weights.
    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected):
        print(f"[ConvRot] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", file=sys.stderr)

    return {
        "int8": sum(1 for b in replaced if plan[b]["format"] == "int8_tensorwise"),
        "int4": sum(1 for b in replaced if plan[b]["format"] == "convrot_w4a4"),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def load_convrot_into_meta_model(
    model: nn.Module,
    checkpoint,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Memory-lean ConvRot load for a model built on the **meta** device.

    This is the ConvRot (int8_tensorwise / convrot_w4a4) counterpart of
    :func:`load_gguf_into_meta_model`. :func:`load_convrot_state_dict` needs a fully
    materialized model, so the whole network exists once as full-precision weights in
    RAM *before* the checkpoint is even read -- i.e. a complete fp/bf16 copy and the
    checkpoint briefly coexist (roughly two copies).

    This variant avoids that. Build the model on meta (zero real allocation), then let
    the compact quantized bytes (plus the small non-quant tensors) be *assigned* straight
    into it with ``load_state_dict(assign=True)``. Only the tiny quantized buffers ever
    hit RAM; the full bf16 weight of each layer is produced lazily on its first forward,
    never all at once at load time. Works on ANY architecture that emits ``.comfy_quant``
    markers (not just SenseNova)::

        with torch.device("meta"):
            model = MyModel(config)                 # zero real memory
        info = load_convrot_into_meta_model(model, "model_int4_convrot.safetensors",
                                            device="cpu", dtype=torch.bfloat16)
        model = model.to(device).eval()             # already on `device`; ready to run

    The quantized buffers keep their integer dtype (int8/int4); ``weight_scale`` is kept
    float32 (not downcast) to preserve de-quant accuracy; plain fp weights (embeddings,
    norms, non-quantized Linears) are cast to ``dtype`` and placed on ``device``. Any keys
    the checkpoint doesn't cover are materialized as uninitialized tensors and reported
    under ``meta_materialized``. Requires torch>=2.1 (load_state_dict assign).
    """
    state_dict = load_any_state_dict(checkpoint, device=device) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_convrot_layers(state_dict)
    if verbose:
        print(f"[ConvRot-meta] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_convrot(model, plan, verbose=verbose, on_meta=True) if plan else []
    active_plan = {b: plan[b] for b in replaced}
    remap_state_dict_keys(state_dict, active_plan, drop_marker=True)
    # also strip markers for planned-but-not-replaced layers to avoid unexpected keys
    for base in plan:
        if base not in active_plan:
            state_dict.pop(f"{base}.comfy_quant", None)

    # Prepare tensors for assign: quant buffers keep dtype (int8/int4); scales stay
    # float32 (don't downcast -> dequant precision); plain fp weights -> dtype; all -> device.
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith((".weight_int8", ".weight_packed")):
            state_dict[k] = v.to(device=device).contiguous()            # keep int8/int4
        elif k.endswith(".weight_scale"):
            state_dict[k] = v.to(device=device, dtype=torch.float32).contiguous()  # keep fp32
        elif v.is_floating_point():
            state_dict[k] = v.to(device=device, dtype=dtype)
        else:
            state_dict[k] = v.to(device=device)

    # assign=True REPLACES the meta params/buffers with the real checkpoint tensors,
    # rather than copy_-ing into pre-allocated full-precision storage -> no 2nd full copy.
    result = model.load_state_dict(state_dict, strict=strict, assign=True)

    # Any params the checkpoint didn't cover are still meta -> give them real storage
    # so the model is runnable; report them so the caller knows what was left uninit.
    leftover = _materialize_leftover_meta(model, device, dtype)

    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected or leftover):
        print(f"[ConvRot-meta] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected, {len(leftover)} meta-materialized", file=sys.stderr)

    return {
        "int8": sum(1 for b in replaced if plan[b]["format"] == "int8_tensorwise"),
        "int4": sum(1 for b in replaced if plan[b]["format"] == "convrot_w4a4"),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "meta_materialized": leftover,
    }


def load_gguf_into_model(
    model: nn.Module,
    checkpoint,
    *,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """One-shot GGUF load: detect -> patch modules -> remap keys -> load_state_dict.

    ``checkpoint`` may be a .gguf path (str) or an already-loaded state dict of
    GGUFParameter / plain tensors. Same contract as :func:`load_convrot_state_dict`:

        info = load_gguf_into_model(model, "model-Q4_K_S.gguf")
        model = model.to(torch.bfloat16).to(device).eval()   # load first, cast later

    The GGUFDequantLinear modules protect their uint8 buffer from the later ``.to()`` and
    de-quantize (then cache bf16) on the first forward.
    """
    _require_gguf()
    state_dict = load_gguf_state_dict(checkpoint) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_gguf_layers(state_dict)
    if verbose:
        print(f"[GGUF] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_gguf(model, plan, verbose=verbose) if plan else []
    active_plan = {b: plan[b] for b in replaced}
    remap_gguf_state_dict_keys(state_dict, active_plan)

    result = model.load_state_dict(state_dict, strict=strict)
    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected):
        print(f"[GGUF] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", file=sys.stderr)

    return {
        "gguf": len(replaced),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _materialize_leftover_meta(model: nn.Module, device, dtype) -> list:
    """Give real storage to any params/buffers still on the meta device.

    After an assign-load some entries may remain on meta (the checkpoint didn't cover
    that key). We can't call ``model.to_empty()`` -- that would wipe the freshly assigned
    real tensors -- so we only touch entries that are STILL meta. Floating params get
    ``dtype``; integer/bool ones keep theirs. Returns their qualified names.
    """
    touched = []
    for mod_name, mod in model.named_modules():
        for pname, p in list(mod._parameters.items()):
            if p is not None and getattr(p, "is_meta", False):
                nd = dtype if p.is_floating_point() else p.dtype
                mod._parameters[pname] = nn.Parameter(
                    torch.empty(p.shape, dtype=nd, device=device),
                    requires_grad=p.requires_grad,
                )
                touched.append(f"{mod_name}.{pname}" if mod_name else pname)
        for bname, b in list(mod._buffers.items()):
            if b is not None and getattr(b, "is_meta", False):
                nd = dtype if b.is_floating_point() else b.dtype
                mod._buffers[bname] = torch.empty(b.shape, dtype=nd, device=device)
                touched.append(f"{mod_name}.{bname}" if mod_name else bname)
    return touched


def load_gguf_into_meta_model(
    model: nn.Module,
    checkpoint,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Memory-lean GGUF load for a model built on the **meta** device.

    :func:`load_gguf_into_model` needs a fully materialized model, so the whole network
    exists once as full-precision weights in RAM *before* the checkpoint is even read --
    i.e. a complete fp/bf16 copy and the GGUF bytes briefly coexist (roughly two copies).

    This variant avoids that. Build the model on meta (no real allocation), then let the
    GGUF bytes (plus the small non-quant tensors) be *assigned* straight into it. Only the
    compact quantized bytes ever hit RAM; the full bf16 weight of each layer is produced
    lazily on its first forward (``weight_cached``), never all at once at load time. This
    mirrors VoxCPM2's ``set_gguf2meta_model`` but is self-contained (no diffusers).

        with torch.device("meta"):
            model = MyModel(config)               # zero real memory
        info = load_gguf_into_meta_model(model, "model-Q4_K.gguf",
                                         device="cpu", dtype=torch.bfloat16)
        model = model.to(device).eval()           # already on `device`; ready to run

    Non-quant params/buffers are cast to ``dtype`` (floats only) and placed on ``device``.
    Any keys the checkpoint doesn't cover are materialized as uninitialized tensors and
    reported under ``meta_materialized``. Requires torch>=2.1 (load_state_dict assign).
    """
    _require_gguf()
    state_dict = load_gguf_state_dict(checkpoint) if isinstance(checkpoint, str) else dict(checkpoint)

    plan = detect_gguf_layers(state_dict)
    if verbose:
        print(f"[GGUF-meta] detected {len(plan)} quantized layer(s) in checkpoint", file=sys.stderr)

    replaced = patch_model_with_gguf(model, plan, verbose=verbose, on_meta=True) if plan else []
    active_plan = {b: plan[b] for b in replaced}
    remap_gguf_state_dict_keys(state_dict, active_plan)

    # Prepare tensors for assign: quant bytes stay uint8; floats -> dtype; all -> device.
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if not isinstance(v, torch.Tensor):
            continue
        if k.endswith(".weight_gguf"):
            state_dict[k] = v.to(device=device).contiguous()
        elif v.is_floating_point():
            state_dict[k] = v.to(device=device, dtype=dtype)
        else:
            state_dict[k] = v.to(device=device)

    # assign=True REPLACES the meta params/buffers with the real checkpoint tensors,
    # rather than copy_-ing into pre-allocated full-precision storage -> no 2nd full copy.
    result = model.load_state_dict(state_dict, strict=strict, assign=True)

    # Any params the checkpoint didn't cover are still meta -> give them real storage
    # so the model is runnable; report them so the caller knows what was left uninit.
    leftover = _materialize_leftover_meta(model, device, dtype)

    _INTERNAL = (".weight_cached", "._lora_A_buf", "._lora_B_buf")
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.endswith(_INTERNAL)]
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if verbose and (missing or unexpected or leftover):
        print(f"[GGUF-meta] load_state_dict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected, {len(leftover)} meta-materialized", file=sys.stderr)

    return {
        "gguf": len(replaced),
        "replaced": replaced,
        "detected": len(plan),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "meta_materialized": leftover,
    }


def _merge_lora_into_bases(model: nn.Module, lora_state_dict: dict, alpha, r, tag: str):
    """Shared LoRA-merge for any _ConvRotBase module (ConvRot int8/int4 or GGUF).

    The LoRA checkpoint may use either key layout (named by the *keys*, not by any
    specific model, so e.g. FLUX/SD/diffusers GGUF LoRA all share the first layout):

      * "down_up" layout : ``"<base>.lora_down.weight"`` / ``"<base>.lora_up.weight"``
        plus an optional ``"<base>.alpha"`` scalar tensor -- this is the standard LoRA
        layout that ``apply_loras_gguf`` / ``set_gguf2meta_model`` expect;
      * "ab" layout      : ``"<base>.lora_A"`` / ``"<base>.lora_B"`` (A/B-style checkpoints,
        e.g. VoxCPM2/ConvRot).

    The delta (scaling * B @ A, scaling = alpha / rank) is stored on the module and merged
    into the de-quantized weight on every forward (see ``_ConvRotBase.forward``).

    The alpha default depends on the key layout (they are NOT interchangeable):
      * "down_up" : alpha read from the file's per-layer ``"<base>.alpha"``, defaulting
        to ``1.0`` when absent (mirrors native ``_prepare_deltas``) -> scaling = 1.0/rank;
      * "ab"      : no per-layer ``.alpha`` key and alpha == rank by convention, so the
        externally supplied ``alpha`` (or ``rank`` if unspecified) is used -> scaling = 1.0.
    An external ``alpha`` only applies to the "ab" layout; for "down_up" the file's
    ``.alpha`` is authoritative (external alpha is ignored there).
    Returns ``(merged_bases, leftover_bases)``.
    """
    base_paths = {}
    for name, m in model.named_modules():
        if isinstance(m, _ConvRotBase):
            base_paths[name] = m
            base_paths[name.replace("._orig_mod.", ".")] = m  # torch.compile wrapper

    # 1) Normalize the LoRA dict into (base -> (lora_A, lora_B, key_format)).
    #    ``key_format`` is derived purely from the KEY SUFFIX that matched -- it is NOT a
    #    field read from the dict, and it names the *key layout*, not any specific model
    #    (so FLUX/SD/diffusers GGUF LoRA all share "down_up", and any A/B-style checkpoint
    #    shares "ab"). The two layouts differ in what "no explicit alpha" means, so we keep
    #    the format only to pick the correct NO-alpha default:
    #      * "down_up" : keys lora_down.weight / lora_up.weight + optional ".alpha". This is
    #          the standard LoRA layout (diffusers/SD/FLUX GGUF, etc.); native
    #          _prepare_deltas uses alpha = 1.0 when there is no "<base>.alpha" key
    #          -> scaling = 1.0/rank.
    #      * "ab"      : keys lora_A / lora_B, no per-layer .alpha. By convention alpha ==
    #          rank here  -> scaling = 1.0.
    lora_map: dict[str, tuple] = {}
    for key in lora_state_dict:
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            up_key = base + ".lora_up.weight"
            if up_key in lora_state_dict and base not in lora_map:
                lora_map[base] = (lora_state_dict[key], lora_state_dict[up_key], "down_up")
        elif key.endswith(".lora_A"):
            base = key[: -len(".lora_A")]
            b_key = base + ".lora_B"
            if b_key in lora_state_dict and base not in lora_map:
                lora_map[base] = (lora_state_dict[key], lora_state_dict[b_key], "ab")

    merged, leftover = [], []
    for base, (lora_A, lora_B, key_format) in lora_map.items():
        mod = base_paths.get(base)
        if mod is None:
            leftover.append(base)
            continue
        rr = r if r is not None else lora_A.shape[0]
        # The alpha VALUE is decided by the LoRA keys themselves:
        #   * if "<base>.alpha" exists in the dict -> use it (the "alpha route", exactly
        #     like native _prepare_deltas: alpha = float(lora_sd[prefix+".alpha"])).
        #   * otherwise fall back to the key-format's default.
        alpha_key = base + ".alpha"
        if alpha_key in lora_state_dict:
            a_scalar = float(lora_state_dict[alpha_key])
        elif key_format == "down_up":
            # Standard down/up layout without an .alpha key: native defaults to 1.0
            # -> scaling 1.0/rank (NOT 1.0, which would over-apply the LoRA / over-expose).
            a_scalar = 1.0
        else:
            # A/B layout: no .alpha key; alpha == rank by convention -> scaling 1.0. An
            # externally supplied alpha/r overrides; else default to rank.
            a_scalar = alpha if alpha is not None else float(rr)
        mod.load_lora_buffer(lora_A, lora_B, a_scalar, rr)
        merged.append(base)
    if merged:
        print(f"[{tag}] merged LoRA into {len(merged)} layer(s); "
              f"{len(leftover)} non-matching key(s) left for nn.Linear.", file=sys.stderr)
    else:
        print(f"[{tag}] WARNING: no LoRA keys matched any de-quant layer "
              f"(scanned {len(lora_state_dict)} LoRA tensors) -- LoRA will NOT be applied.",
              file=sys.stderr)
    return merged, leftover


def load_convrot_lora(model: nn.Module, lora_state_dict: dict, alpha=None, r=None):
    """Merge LoRA weights into ConvRot*Linear layers (de-quant-time merge).

    ``lora_state_dict`` maps ``"<module_path>.lora_A"`` / ``".lora_B"`` -> tensor (the same
    key convention VoxCPM2 uses). For every matching de-quant module the LoRA is stored as a
    buffer and merged into the de-quantized weight on each forward. Modules whose path does
    NOT match are reported in ``leftover`` so the caller can apply plain ``nn.Linear`` LoRA.

    Returns ``(merged_bases, leftover_bases)``.
    """
    return _merge_lora_into_bases(model, lora_state_dict, alpha, r, tag="ConvRot")


def load_gguf_lora(model: nn.Module, lora_state_dict: dict, alpha=None, r=None):
    """Merge LoRA into GGUFDequantLinear layers (de-quant-time merge).

    Identical mechanism to :func:`load_convrot_lora` (both share the _ConvRotBase LoRA
    path); provided as a distinct entry point for GGUF-loaded models.
    Returns ``(merged_bases, leftover_bases)``.
    """
    return _merge_lora_into_bases(model, lora_state_dict, alpha, r, tag="GGUF")


__all__ = [
    "ConvRotInt8Linear",
    "ConvRotInt4Linear",
    "GGUFDequantLinear",
    "detect_convrot_layers",
    "patch_model_with_convrot",
    "remap_state_dict_keys",
    "detect_gguf_layers",
    "patch_model_with_gguf",
    "remap_gguf_state_dict_keys",
    "load_any_state_dict",
    "load_gguf_state_dict",
    "load_convrot_state_dict",
    "load_convrot_into_meta_model",
    "load_gguf_into_model",
    "load_gguf_into_meta_model",
    "load_convrot_lora",
    "load_gguf_lora",
]
