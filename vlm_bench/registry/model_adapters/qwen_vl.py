from typing import Any, Dict, List
from vlm_bench.registry.model_adapters.base import ModelAdapter
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class QwenVLAdapter(ModelAdapter):
    family = "qwen-vl"

    def load(self):
        self._engine = self.model_cfg.get("__shared_engine", None)
        self.vision_args = self.model_cfg.get("vision_args", {"max_pixels": 1024*1024})

    def default_decode(self) -> Dict[str, Any]:
        d = super().default_decode()
        d["max_new_tokens"] = 768
        return d

    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        if self._engine is None:
            raise RuntimeError("Engine not initialized")
        gens: List[Generation] = []
        for s in batch:
            prompt = self._render_prompt(s)
            text = self._engine.generate(s.media, prompt, decode)
            gens.append(Generation(id=s.id, text=text))
        return gens

    def _render_prompt(self, s: Sample) -> str:
        sys_prompt = self.model_cfg.get("system_prompt", "You are a helpful vision-language assistant.")
        return f"{sys_prompt}\n\n{s.prompt}"
