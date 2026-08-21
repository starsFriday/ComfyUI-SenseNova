from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .quantized_nodes import SenseNovaU15ImageEdit, SenseNovaU15ModelLoader, SenseNovaU15TextToImage


class SenseNova_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaU15ModelLoader,
            SenseNovaU15TextToImage,
            SenseNovaU15ImageEdit,
        ]   

async def comfy_entrypoint() -> SenseNova_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return SenseNova_SM_Extension()
