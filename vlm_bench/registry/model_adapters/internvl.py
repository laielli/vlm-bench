from typing import Any, Dict, List
from vlm_bench.registry.model_adapters.base import ModelAdapter
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class InternVLAdapter(ModelAdapter):
    family = "internvl"

    def load(self):
        self.vision_args = self.model_cfg.get("vision_args", {"short_side": 448})
        self._engine = self.model_cfg.get("__shared_engine", None)

    def default_decode(self) -> Dict[str, Any]:
        d = super().default_decode()
        d["max_new_tokens"] = 768
        return d

    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        if self._engine is None:
            raise RuntimeError("Engine not initialized")
        gens = []
        for s in batch:
            prompt = s.prompt
            text = self._engine.generate(s.media, prompt, decode)
            gens.append(Generation(id=s.id, text=text))
        return gens
