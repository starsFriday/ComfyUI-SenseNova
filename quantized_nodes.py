import os

import torch
from comfy_api.latest import io
from safetensors import safe_open
from tqdm.auto import tqdm

import comfy.utils
import folder_paths

from .node_utils import clear_comfyui_cache, tensor2pillist
from .SenseNova.examples.editing.inference import load_sensenova_model


SenseNovaU15Model = io.Custom("SENSENOVA_U15_MODEL")
MAX_SEED = 2**32 - 1
NODE_PATH = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MAX_REFERENCE_IMAGES = 64


def _checkpoint_format(path):
    if not path.lower().endswith(".safetensors"):
        raise ValueError("SenseNova U1.5 Model Loader only accepts .safetensors checkpoints.")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        if any(key.endswith(".comfy_quant") for key in keys):
            return "int8_convrot"
        if any(key.endswith("scaled_fp8") for key in keys) and any(key.endswith(".scale_weight") for key in keys):
            return "fp8_scaled"
        bf16_signature = {
            "fm_modules.fm_head.conv1.weight",
            "language_model.model.embed_tokens.weight",
        }
        if bf16_signature.issubset(keys) and all(checkpoint.get_slice(key).get_dtype() in {"BF16", "F32"} for key in keys):
            return "bf16"
    raise ValueError("Checkpoint is not a supported SenseNova U1.5 BF16, INT8 ConvRot, or scaled FP8 file.")


def _validate_model_load(model, expected_format):
    if model.quantization_format != expected_format:
        raise ValueError(f"Expected {expected_format}, loaded {model.quantization_format or 'unknown'}.")
    info = model.quant_load_info
    if expected_format == "bf16":
        count = info.get("loaded_tensors", 0)
        missing = info.get("missing_keys", [])
        unexpected = info.get("unexpected_keys", [])
        if missing or unexpected:
            raise ValueError(f"BF16 checkpoint/config mismatch: {len(missing)} missing and {len(unexpected)} unexpected tensors.")
    else:
        count = info.get("int8", 0) if expected_format == "int8_convrot" else info.get("fp8", 0)
    if count == 0:
        raise ValueError(f"No {expected_format} tensors were loaded from the checkpoint.")
    leftover = info.get("meta_materialized", [])
    if leftover:
        names = ", ".join(leftover[:5])
        raise ValueError(f"Checkpoint/config mismatch left {len(leftover)} tensors uninitialized: {names}")


def _run_model(model, low_vram, callback):
    clear_comfyui_cache()
    if low_vram:
        return callback(1)

    model.model.to(DEVICE)
    try:
        return callback(None)
    finally:
        model.model.to("cpu")
        clear_comfyui_cache()


def _image_output(image):
    return (image.float() * 0.5 + 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu()


def _image_size(width, height):
    if not 256 <= width <= 4096 or not 256 <= height <= 4096:
        raise ValueError("SenseNova width and height must be between 256 and 4096 pixels.")
    if width % 32 or height % 32:
        raise ValueError("SenseNova width and height must both be multiples of 32 pixels.")
    return width, height


def _progress_callback(steps):
    progress = comfy.utils.ProgressBar(steps)
    terminal = tqdm(total=steps, desc="SenseNova sampling", disable=not comfy.utils.PROGRESS_BAR_ENABLED)

    def update(value, total):
        progress.update_absolute(value, total)
        terminal.total = total
        terminal.update(value - terminal.n)
        if value >= total:
            terminal.close()

    return update


def _reference_images(inputs):
    inputs = inputs or {}
    images = [inputs[f"image_{index}"] for index in range(MAX_REFERENCE_IMAGES) if inputs.get(f"image_{index}") is not None]
    image_count = sum(image.shape[0] for image in images)
    if image_count == 0:
        raise ValueError("SenseNova image editing requires at least one reference image.")
    if image_count > MAX_REFERENCE_IMAGES:
        raise ValueError(f"At most {MAX_REFERENCE_IMAGES} reference images are supported, including image batches.")
    references = []
    for image in images:
        references.extend(tensor2pillist(image))
    return references


class SenseNovaU15ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15ModelLoader",
            display_name="SenseNova U1.5 Model Loader",
            category="SenseNova/U1.5",
            description="Loads a pruned BF16, scaled-FP8, or ComfyUI INT8 ConvRot single-file checkpoint.",
            inputs=[
                io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("diffusion_models"),
                               tooltip="Put the supported BF16, scaled-FP8, or INT8 ConvRot .safetensors file in models/diffusion_models."),
                io.Combo.Input("attention", options=["auto", "sdpa", "flash"], default="auto"),
            ],
            outputs=[SenseNovaU15Model.Output(display_name="SenseNova U1.5 model")],
        )

    @classmethod
    def execute(cls, checkpoint, attention):
        if not torch.cuda.is_available():
            raise RuntimeError("SenseNova U1.5 inference requires an NVIDIA CUDA GPU.")
        clear_comfyui_cache()
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        expected_format = _checkpoint_format(path)
        config_repo = os.path.join(NODE_PATH, "SenseNova-U1.5-8B-MoT")
        if not os.path.isdir(config_repo):
            raise RuntimeError("SenseNova-U1.5-8B-MoT config/tokenizer files are missing from this custom node.")
        model = load_sensenova_model(path, DEVICE, attention, config_repo)
        _validate_model_load(model, expected_format)
        return io.NodeOutput(model)


class SenseNovaU15TextToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15TextToImage",
            display_name="SenseNova U1.5 Text to Image",
            category="SenseNova/U1.5",
            description="Native SenseNova pixel-space flow-matching generation for the pruned checkpoint.",
            inputs=[
                SenseNovaU15Model.Input("model"),
                io.String.Input("prompt", multiline=True, default="A cinematic mountain lake at sunrise, realistic photography."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=100, step=1),
                io.Int.Input("width", default=2048, min=256, max=4096, step=32),
                io.Int.Input("height", default=2048, min=256, max=4096, step=32),
                io.Float.Input("cfg", default=4.0, min=1.0, max=10.0, step=0.1),
                io.Combo.Input("cfg_norm", options=["none", "global", "channel", "cfg_zero_star"], default="none"),
                io.Float.Input("timestep_shift", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("batch_size", default=1, min=1, max=8, step=1),
                io.Boolean.Input("low_vram", default=True,
                                 tooltip="Streams one transformer layer at a time. Disable only when the full model and KV cache fit in VRAM."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, steps, width, height, cfg, cfg_norm, timestep_shift, batch_size, low_vram):
        if model.pruned_lm_head is False:
            raise ValueError("This node is intended for the pruned SenseNova U1.5 T2I/edit checkpoint.")
        image_size = _image_size(width, height)

        def generate(prefetch_count):
            return model.generate(
                prompt,
                image_size=image_size,
                cfg_scale=cfg,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                num_steps=steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=False,
                streaming_prefetch_count=prefetch_count,
                progress_callback=_progress_callback(steps),
            )

        return io.NodeOutput(_run_model(model, low_vram, generate))


class SenseNovaU15ImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SenseNovaU15ImageEdit",
            display_name="SenseNova U1.5 Image Edit",
            category="SenseNova/U1.5",
            description="Edits one or more reference images with SenseNova's native three-branch guidance.",
            inputs=[
                SenseNovaU15Model.Input("model"),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image", tooltip="Reference images are passed to SenseNova in socket and batch order."),
                        prefix="image_",
                        min=1,
                        max=MAX_REFERENCE_IMAGES,
                    ),
                    tooltip="One or more ordered references. Connecting the last socket reveals the next one.",
                ),
                io.String.Input("prompt", multiline=True,
                                default="Change the jacket to cobalt blue. Preserve the face, pose, background, lighting, and framing."),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=100, step=1),
                io.Int.Input("width", default=2048, min=256, max=4096, step=32),
                io.Int.Input("height", default=2048, min=256, max=4096, step=32),
                io.Float.Input("cfg", default=4.0, min=1.0, max=10.0, step=0.1),
                io.Float.Input("image_cfg", default=1.0, min=1.0, max=10.0, step=0.1),
                io.Combo.Input("cfg_norm", options=["none", "global", "channel"], default="none"),
                io.Float.Input("timestep_shift", default=3.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("batch_size", default=1, min=1, max=8, step=1),
                io.Boolean.Input("low_vram", default=True,
                                 tooltip="Streams one transformer layer at a time. Recommended for 24 GB GPUs."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, model, images, prompt, seed, steps, width, height, cfg, image_cfg, cfg_norm, timestep_shift,
                batch_size, low_vram):
        if model.pruned_lm_head is False:
            raise ValueError("This node is intended for the pruned SenseNova U1.5 T2I/edit checkpoint.")
        references = _reference_images(images)
        image_size = _image_size(width, height)

        def edit(prefetch_count):
            output = model.edit(
                prompt,
                references,
                image_size=image_size,
                cfg_scale=cfg,
                img_cfg_scale=image_cfg,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                num_steps=steps,
                batch_size=batch_size,
                think_mode=False,
                seed=seed,
                streaming_prefetch_count=prefetch_count,
                progress_callback=_progress_callback(steps),
            )
            return _image_output(output)

        return io.NodeOutput(_run_model(model, low_vram, edit))
