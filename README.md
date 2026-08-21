# ComfyUI-SenseNova

ComfyUI custom nodes for SenseNova U1.5. The project supports text-to-image generation and single- or multi-reference image editing with pruned, single-file BF16, scaled-FP8, and INT8 ConvRot checkpoints.

## Installation

Run the following commands from `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/starsFriday/ComfyUI-SenseNova.git
cd ComfyUI-SenseNova
pip install -r requirements.txt
```

Restart ComfyUI after installation. The final SenseNova U1.5 configuration and tokenizer metadata are bundled with the nodes, so no separate configuration repository is required.

## Models

Download a supported checkpoint from [joyfox/SenseNova-U1.5-8B-MoT-FP8](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8) and place it in `ComfyUI/models/diffusion_models/`:

```text
ComfyUI/models/diffusion_models/
├── SenseNova-U1.5-8B-MoT-pruned-bf16.safetensors
├── SenseNova-U1.5-8B-MoT-pruned-fp8_scaled.safetensors
└── SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors
```

The loader validates the checkpoint format, loaded tensors, and whether any weights remain uninitialized. The current inference path requires an NVIDIA CUDA GPU. BF16 needs substantially more system memory and GPU memory than the quantized variants; enable `low_vram` unless the complete model and KV cache fit in VRAM.

## Nodes

- `SenseNova U1.5 Model Loader` loads a supported single-file checkpoint from `models/diffusion_models`.
- `SenseNova U1.5 Text to Image` runs native SenseNova text-to-image sampling.
- `SenseNova U1.5 Image Edit` runs native single- or multi-reference image editing.

Both generation nodes expose independent `width` and `height` controls. The default resolution is `2048 × 2048`, and the supported range is 256–4096 pixels. Because the image grid uses `patch_size 16 × merge_size 2`, both dimensions must be divisible by 32. This is validated again when the node executes.

`Image Edit` uses an `io.Autogrow` IMAGE input. Connecting `image_0` reveals `image_1`, and additional sockets appear as needed, up to 64 reference images. The U1.5 implementation allocates at least `512²` pixels to each image while limiting all references to a total `4096²` pixel budget. At 64 images these limits are equal, so additional references are not valid. IMAGE batches are expanded in socket order and then batch order, and every expanded image counts toward the limit. Use `Image-1`, `Image-2`, and so on in the prompt to refer to specific images.

Sampling progress is displayed with ComfyUI's native progress bar above the node. The terminal also shows completed steps, elapsed time, estimated remaining time, and iteration speed, matching the standard sampler experience. `low_vram` is enabled by default and streams the Transformer one layer at a time. Keep it enabled on 24 GB GPUs. Disable it only when the full model and KV caches fit in VRAM.

The pruned checkpoints remove `language_model.lm_head`, which is used only for text output. These nodes therefore support text-to-image generation and image editing, but not Think mode, VQA text output, or interleaved text/image output. Recommended starting settings are `steps=50`, `cfg=4.0`, and `timestep_shift=3.0`.

## Example workflows

- `example_workflows/U15_t2i.json`
- `example_workflows/U15_edit.json`

The edit workflow demonstrates two dynamic reference-image inputs. After importing it, select local images in both `Load Image` nodes.

## Why there is no KSampler connection

SenseNova first runs an autoregressive text/image prefix and builds a per-layer KV cache, then performs native pixel-space Flow Matching. Image editing additionally maintains condition, image-condition, and unconditional caches. ComfyUI's standard `MODEL + CONDITIONING + LATENT` KSampler contract cannot preserve this state or the three-branch editing semantics, so this project keeps SenseNova's native Euler sampling path.

## Credits and license

The model and original inference implementation come from [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1). This project adds the ComfyUI integration and reuses ComfyUI and Comfy Kitchen model-management and quantized inference operations.

The project code is licensed under the [Apache License 2.0](LICENSE). Model weights remain subject to the license published with the corresponding model repository.
