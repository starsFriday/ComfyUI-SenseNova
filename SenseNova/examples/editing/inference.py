from __future__ import annotations

import gc
from contextlib import AbstractContextManager
from typing import Sequence

import torch
from accelerate import init_empty_weights
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..utils import _streaming_model
from ...src.sensenova_u1.models.neo_unify.modeling_qwen3 import set_attn_backend


NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def _to_tensor(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu()


class SenseNovaU1Editing:
    def __init__(
        self,
        checkpoint: str,
        config_repo: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(config_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(config_repo)
        self.model = None
        self.pruned_lm_head = False
        self.quantization_format = None
        self.quant_load_info = {}

    def load(self) -> None:
        if self.model is not None:
            return

        with init_empty_weights():
            self.model = AutoModel.from_config(self.config)

        state_dict = load_file(self.checkpoint)
        self.pruned_lm_head = "language_model.lm_head.weight" not in state_dict
        if self.pruned_lm_head:
            self.model.language_model.lm_head = torch.nn.Identity()

        has_fp8_scales = any(key.endswith(".scale_weight") for key in state_dict)
        has_convrot = any(key.endswith(".comfy_quant") for key in state_dict)
        if has_fp8_scales:
            from ...fp8_scaled_loader import load_fp8_scaled_into_meta_model

            info = load_fp8_scaled_into_meta_model(self.model, state_dict, use_scaled_mm=True)
            self.quantization_format = "fp8_scaled"
        elif has_convrot:
            from ...convrot_loader import load_convrot_into_meta_model

            info = load_convrot_into_meta_model(self.model, state_dict)
            self.quantization_format = "int8_convrot"
        else:
            loaded_tensors = len(state_dict)
            for key, value in state_dict.items():
                if value.is_floating_point():
                    state_dict[key] = value.to(dtype=self.dtype)
            result = self.model.load_state_dict(state_dict, strict=False, assign=True)
            meta_materialized = [name for name, value in self.model.named_parameters() if value.is_meta]
            meta_materialized.extend(name for name, value in self.model.named_buffers() if value.is_meta)
            info = {
                "loaded_tensors": loaded_tensors,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "meta_materialized": meta_materialized,
            }
            self.quantization_format = "bf16"

        self.quant_load_info = info
        self.model.eval()
        del state_dict
        gc.collect()

    def _model_ctx(self, prefetch_count: int) -> AbstractContextManager:
        return _streaming_model(
            self.model,
            layers_attr="language_model.model.layers",
            target_device=self.device,
            prefetch_count=prefetch_count,
        )

    def edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        image_size: tuple[int, int],
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        think_mode: bool = False,
        seed: int = 0,
        streaming_prefetch_count: int | None = 1,
        progress_callback=None,
    ) -> torch.Tensor:
        if streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as model:
                return model.it2i_generate(
                    self.tokenizer,
                    prompt,
                    list(images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    think_mode=think_mode,
                    seed=seed,
                    progress_callback=progress_callback,
                )

        return self.model.it2i_generate(
            self.tokenizer,
            prompt,
            list(images),
            image_size=image_size,
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            batch_size=batch_size,
            think_mode=think_mode,
            seed=seed,
            progress_callback=progress_callback,
        )

    def generate(
        self,
        prompt: str,
        image_size: tuple[int, int],
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        think_mode: bool = False,
        streaming_prefetch_count: int | None = 1,
        progress_callback=None,
    ) -> torch.Tensor:
        if streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as model:
                output = model.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seed=seed,
                    think_mode=think_mode,
                    progress_callback=progress_callback,
                )
        else:
            output = self.model.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
                progress_callback=progress_callback,
            )
        return _to_tensor(output)


def load_sensenova_model(
    checkpoint: str,
    device: torch.device,
    attention: str,
    config_repo: str,
    dtype: torch.dtype = torch.bfloat16,
) -> SenseNovaU1Editing:
    set_attn_backend(attention)
    engine = SenseNovaU1Editing(checkpoint, config_repo, device, dtype)
    engine.load()
    return engine
