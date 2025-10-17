from typing import List, Any, Dict
from vlm_bench.registry.model_adapters.base import ModelAdapter
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class NativeAdapter(ModelAdapter):
    family = "native"

    def load(self):
        # Implement with transformers/vLLM stack when ready.
        pass

    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        raise NotImplementedError("Implement me once load() is done")
