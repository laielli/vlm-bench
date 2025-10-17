from __future__ import annotations
from typing import Any, Dict, List
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class ModelAdapter:
    family: str = "base"

    def __init__(self, model_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self._model = None

    def load(self):
        raise NotImplementedError

    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        raise NotImplementedError

    def default_decode(self) -> Dict[str, Any]:
        return {"temperature": 0.0, "max_new_tokens": 512, "stop": []}
