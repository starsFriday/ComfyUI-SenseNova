from __future__ import annotations
import functools
import itertools
import logging
from typing import Any
import torch.nn.functional as F
import torch
from torch import nn
from .src.sensenova_u1.models.neo_unify.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
logger = logging.getLogger(__name__)

class ExpertOffloadMoEBlock(nn.Module):
    """
    同步卸载，支持自定义缓存容量：在 GPU 上最多同时保留 `cache_capacity` 个专家。
    gate 常驻 GPU，专家按需同步加载/卸载，采用 LRU 淘汰策略。
    """
    def __init__(self, moe_block, target_device: torch.device, cache_capacity: int = 1):
        super().__init__()
        self.moe_block = moe_block
        self.target_device = target_device
        self.cache_capacity = max(1, cache_capacity)  # 至少为1
        # gate 常驻 GPU
        self.moe_block.gate.to(target_device)
        self._resident_experts: list[int] = []

    # ----- 代理原始属性 -----
    @property
    def num_experts(self):
        return self.moe_block.num_experts

    @property
    def top_k(self):
        return self.moe_block.top_k

    @property
    def norm_topk_prob(self):
        return self.moe_block.norm_topk_prob

    # ----- LRU 管理 -----
    def _touch(self, eid: int):
        """将专家 eid 移到 LRU 队列末尾（最近使用）。"""
        if eid in self._resident_experts:
            self._resident_experts.remove(eid)
        self._resident_experts.append(eid)

    def _evict_lru(self):
        """如果驻留数量超过 cache_capacity，卸载最久未使用的专家。"""
        while len(self._resident_experts) >= self.cache_capacity:
            victim = self._resident_experts.pop(0) # 最久未使用
            self._unload_expert(victim)

    def _load_expert(self, eid: int):
        """确保专家 eid 驻留在 GPU。若缓存已满，先淘汰最久未使用的。"""
        if eid in self._resident_experts:
            self._touch(eid)  # 命中，只更新时间戳
            return

        # 淘汰直至有空间
        self._evict_lru()

        # 加载新专家
        expert_module = self.moe_block.experts[eid]
        for param in itertools.chain(expert_module.parameters(), expert_module.buffers()):
            param.data = param.data.to(self.target_device)
        self._resident_experts.append(eid)

    def _unload_expert(self, eid: int):
        """将指定专家从 GPU 移回 CPU。"""
        expert_module = self.moe_block.experts[eid]
        for param in itertools.chain(expert_module.parameters(), expert_module.buffers()):
            param.data = param.data.to('cpu')
      
    def _unload_all(self):
        """卸载所有驻留专家，释放显存。"""
        for eid in list(self._resident_experts):
            self._unload_expert(eid)
        self._resident_experts.clear()

   
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.moe_block.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        active_experts = selected_experts.unique().tolist()

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        output = torch.zeros_like(flat)
       
        for eid in active_experts:
            
            self._load_expert(eid)

            idx, top_x = torch.where(expert_mask[eid])
            if top_x.numel() == 0:
                continue

            expert_module = self.moe_block.experts[eid]
            current_state = flat.index_select(0, top_x)
            expert_out = expert_module(current_state) * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, expert_out.to(flat.dtype))


        # ---- forward 结束，清空所有驻留专家（可选） ----
        #self._unload_all()

        return output.view(*orig_shape)

class ExpertStreamingWrapper(nn.Module):
    """
    同步版模型包装器：替换所有 MoE 块为 ExpertOffloadMoEBlock，
    将非 MoE 参数移到 GPU，MoE 专家按需同步加载。
    """

    def __init__(self, model: nn.Module, target_device: torch.device,cache_capacity: int = 1):
        super().__init__()
        self._model = model
        self._target_device = target_device
        self._cache_capacity = cache_capacity
        self._replace_map: dict[int, tuple[nn.Module, str, nn.Module]] = {}  


        expert_param_ids: set[int] = set()
        for module in model.modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock):
                for expert in module.experts:
                    for p in expert.parameters():
                        expert_param_ids.add(id(p))
                    for b in expert.buffers():
                        expert_param_ids.add(id(b))
                for p in module.gate.parameters():
                    expert_param_ids.add(id(p))
                for b in module.gate.buffers():
                    expert_param_ids.add(id(b))


        self._replace_moe_blocks(model, target_device)

        for p in model.parameters():
            if id(p) not in expert_param_ids:
                p.data = p.data.to(target_device)
        for b in model.buffers():
            if id(b) not in expert_param_ids:
                b.data = b.data.to(target_device)

    def _replace_moe_blocks(self, module: nn.Module, target_device: torch.device):
        replacements = []
        for name, child in module.named_children():
            if isinstance(child, Qwen3MoeSparseMoeBlock):
                replacements.append((name, child))
            else:
                self._replace_moe_blocks(child, target_device)
        for name, original in replacements:
            wrapper = ExpertOffloadMoEBlock(original, target_device, self._cache_capacity)
            setattr(module, name, wrapper)
            self._replace_map[id(wrapper)] = (module, name, original)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def teardown(self):
        for wrapper_id, (parent, name, original) in self._replace_map.items():
            setattr(parent, name, original)
        self._model.to('cpu')
        torch.cuda.synchronize(self._target_device)
        self._replace_map.clear()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
        

# edit from LayerStreamingWrapper from https://github.com/Lightricks/LTX-2

class _SimpleLayerStore:
    """简化版层存储，支持按需加载和立即释放"""

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        # 保留CPU端的原始参数引用
        self._cpu_params: list[dict[str, torch.Tensor]] = []
        for layer in layers:
            cpu_copy = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                if tensor is None:
                    continue
                cpu_copy[name] = tensor.data.cpu()  # 保留在CPU上
            self._cpu_params.append(cpu_copy)

    def load_layer_to_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层加载到GPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name].to(self.target_device)

    def unload_layer_from_gpu(self, idx: int, layer: nn.Module) -> None:
        """将指定层从GPU卸载回CPU"""
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            if param is None:
                continue
            if name in self._cpu_params[idx]:
                param.data = self._cpu_params[idx][name]  # 恢复为CPU副本

def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj

class SimpleLayerStreamingWrapper(nn.Module):
    """简化版层流式处理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        active_count: int = 1,  # 同时激活的层数量
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._active_count = active_count
        self._store = _SimpleLayerStore(self._layers, self._target_device)
        self._hook_handles = []
        
        # 将非层参数移到GPU
        self._move_non_layer_params_to_gpu()
        
        # 注册钩子
        self._register_simple_hooks()
    
    def _move_non_layer_params_to_gpu(self) -> None:
        """移动非层参数到GPU"""
        layer_tensor_ids = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)
    
    def _register_simple_hooks(self) -> None:
        """注册简单的加载/释放钩子"""
        idx_map = {id(layer): idx for idx, layer in enumerate(self._layers)}
        
        def _pre_hook(module: nn.Module, input, *, idx: int):
            # 加载当前层到GPU
            self._store.load_layer_to_gpu(idx, module)
            # 记录流，防止内存被提前回收
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(torch.cuda.current_stream(self._target_device))
        
        def _post_hook(module: nn.Module, input, output, *, idx: int):
            # 处理完后立即将层移回CPU
            self._store.unload_layer_from_gpu(idx, module)
        
        for layer in self._layers:
            idx = idx_map[id(layer)]
            self._hook_handles.append(layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx)))
            self._hook_handles.append(layer.register_forward_hook(functools.partial(_post_hook, idx=idx)))

    def teardown(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._model.to("cpu")
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """代理属性访问到原始模型"""
        try:
            # 首先尝试从包装器自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果失败，则从原始模型获取
            return getattr(self._model, name)
class _LayerStore:
    """Manages CPU-pinned copies of layer parameters/buffers.
    Tracks which layers currently reside on GPU so the prefetcher and evictor
    can make correct decisions.
    """

    def __init__(self, layers: nn.ModuleList, target_device: torch.device) -> None:
        self.target_device = target_device
        self.num_layers = len(layers)

        # CPU-pinned copies keyed by (layer_idx, param_name)
        self._pinned: list[dict[str, torch.Tensor]] = []
        self._on_gpu: set[int] = set()

        for layer in layers:
            pinned: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                pinned_tensor = tensor.data.pin_memory()
                tensor.data = pinned_tensor
                pinned[name] = pinned_tensor
            self._pinned.append(pinned)

    def _check_idx(self, idx: int) -> None:
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(f"Layer index {idx} out of range [0, {self.num_layers})")

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def move_to_gpu(self, idx: int, layer: nn.Module, *, non_blocking: bool = False) -> None:
        """Move layer *idx* parameters from pinned CPU to *target_device*."""
        self._check_idx(idx)
        if idx in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name].to(self.target_device, non_blocking=non_blocking)
        self._on_gpu.add(idx)

    def evict_to_cpu(self, idx: int, layer: nn.Module) -> None:
        """Swap layer *idx* parameters back to their pinned CPU copies."""
        self._check_idx(idx)
        if idx not in self._on_gpu:
            return
        pinned = self._pinned[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = pinned[name]
        self._on_gpu.discard(idx)

    def cleanup(self) -> None:
        """Release all pinned memory references.
        After this call, the pinned tensors can be garbage-collected once
        the layer parameters (which still reference them via ``.data``) are
        also released (e.g. via ``.to("meta")``).
        """
        for pinned_dict in self._pinned:
            pinned_dict.clear()
        self._pinned.clear()


class _AsyncPrefetcher:
    """Issues H2D transfers on a dedicated CUDA stream.
    Uses per-layer CUDA events so that the compute stream only waits for the
    specific layer it needs, not all pending transfers.
    """

    def __init__(self, store: _LayerStore, layers: nn.ModuleList) -> None:
        self._store = store
        self._layers = layers
        self._stream = torch.cuda.Stream(device=store.target_device)
        self._events: dict[int, torch.cuda.Event] = {}

    def prefetch(self, idx: int) -> None:
        """Begin async transfer of layer *idx* to GPU (no-op if already there)."""
        if self._store.is_on_gpu(idx) or idx in self._events:
            return
        with torch.cuda.stream(self._stream):
            self._store.move_to_gpu(idx, self._layers[idx], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            self._events[idx] = event

    def wait(self, idx: int) -> None:
        """Block the compute stream until layer *idx* transfer is complete."""
        event = self._events.pop(idx, None)
        if event is not None:
            torch.cuda.current_stream(self._store.target_device).wait_event(event)

    def cleanup(self) -> None:
        """Drain pending work and release CUDA stream/event resources."""
        self._events.clear()
        self._stream = None
        self._layers = None
        self._store = None


class LayerStreamingWrapper(nn.Module):
    """Wraps a model to stream its sequential layers between CPU and GPU.
    Each layer is evicted immediately after its forward completes, and
    prefetch wraps around using modular indexing so the end of one forward
    pass prepares early layers for the next.
    Parameters
    ----------
    model:
        The model to wrap, with all parameters on **CPU**.
    layers_attr:
        Dotted attribute path to the ``nn.ModuleList`` of sequential layers
        (e.g. ``"transformer_blocks"`` or ``"model.language_model.layers"``).
    target_device:
        The GPU device to use for compute.
    prefetch_count:
        How many layers ahead to prefetch.  The maximum number of layers on
        GPU at once is ``1 + prefetch_count``.  Must be >= 1.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
        prefetch_count: int = 2,
    ) -> None:
        if prefetch_count < 1:
            raise ValueError("prefetch_count must be >= 1")
        super().__init__()
        # Store the wrapped model as a submodule so parameters are discoverable.
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        # Clamp: no point prefetching more than num_layers - 1 (the rest are evicted).
        self._prefetch_count = min(prefetch_count, len(self._layers) - 1)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._setup()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------


    def _setup(self) -> None:
        # 1. Build the pinned CPU store (copies all layer tensors to pinned memory).
        self._store = _LayerStore(self._layers, self._target_device)

        # 2. Move all NON-layer params/buffers to GPU.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        # 3. Pre-load the first (1 + prefetch_count) layers synchronously.
        for idx in range(min(self._prefetch_count + 1, len(self._layers))):
            self._store.move_to_gpu(idx, self._layers[idx])

        # 4. Create the async prefetcher and register hooks.
        self._prefetcher = _AsyncPrefetcher(self._store, self._layers)
        self._register_hooks()


    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
        num_layers = len(self._layers)

        def _pre_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Wait only for THIS layer's H2D transfer (not all pending ones).
            self._prefetcher.wait(idx)
            if not self._store.is_on_gpu(idx):
                self._store.move_to_gpu(idx, module)

            # Record that the compute stream will read these weight tensors.
            # They were allocated on the prefetch stream, so without this the
            # caching allocator would allow the prefetch stream to reuse their
            # memory immediately after eviction — even if the compute kernel
            # that reads them hasn't finished yet.
            compute_stream = torch.cuda.current_stream(self._target_device)
            for param in itertools.chain(module.parameters(), module.buffers()):
                param.data.record_stream(compute_stream)

            # Kick off prefetch for upcoming layers (wraps around for next pass).
            for offset in range(1, self._prefetch_count + 1):
                self._prefetcher.prefetch((idx + offset) % num_layers)

        def _post_hook(
            module: nn.Module,
            _args: Any,  # noqa: ANN401
            _output: Any,  # noqa: ANN401
            *,
            idx: int,
        ) -> None:
            # Evict this layer immediately — its computation is done.
            self._store.evict_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        """Remove hooks, release pinned memory, and move parameters back to CPU.
        After this call the wrapper is inert: hooks are removed, the prefetch
        stream is drained and destroyed, all parameters reside on regular
        (non-pinned) CPU memory, and the ``_LayerStore`` pinned-tensor cache is
        cleared.  Callers should still follow up with ``.to("meta")`` to release
        the CPU copies if the model is no longer needed.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Drain all in-flight async H2D copies, then release stream resources.
        # Without the synchronize, clearing the stream/events can trigger
        # use-after-free at the CUDA driver level.
        torch.cuda.synchronize(device=self._target_device)
        if self._prefetcher is not None:
            self._prefetcher.cleanup()
            self._prefetcher = None

        # Move everything to CPU.
        for idx, layer in enumerate(self._layers):
            self._store.evict_to_cpu(idx, layer)

        for p in self._model.parameters():
            p.data = p.data.to("cpu")
        for b in self._model.buffers():
            b.data = b.data.to("cpu")

        # Release pinned memory.  After evict_to_cpu() the layer parameters
        # still reference the pinned tensors (since .to("cpu") on a pinned
        # tensor is a no-op).  The caller is expected to follow up with
        # .to("meta") to drop the param refs; cleanup() drops the store's refs.
        self._store.cleanup()

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model.
        This allows calling methods like ``encode()`` on a wrapped
        GemmaTextEncoder without the caller needing to know about the wrapper.
        ``nn.Module.__getattr__`` is only called when normal attribute lookup
        fails, so ``_model``, ``_store``, etc. are found first via ``__dict__``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
