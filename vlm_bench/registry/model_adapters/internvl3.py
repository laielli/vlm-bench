from typing import Any, Dict, List
from vlm_bench.registry.model_adapters.base import ModelAdapter
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class InternVL3Adapter(ModelAdapter):
    """Adapter for InternVL3 family.
    - Expects the shared engine to implement `generate_video(media, prompt, decode, model_cfg)`.
    - Supports video QA (mp4 path or list of frames).
    """
    family = "internvl3"

    def load(self):
        self._engine = self.model_cfg.get("__shared_engine")
        if self._engine is None:
            raise RuntimeError("Shared engine not provided. Ensure CLI sets mdl_cfg['__shared_engine'] = engine.")
        self.video_cfg = self.model_cfg.get("video", {}) or {}
        self.vision_args = self.model_cfg.get("vision_args", {}) or {}

    def default_decode(self) -> Dict[str, Any]:
        d = super().default_decode()
        d["max_new_tokens"] = 1024   # InternVL3 long-answer headroom
        return d

    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        gens: List[Generation] = []
        for s in batch:
            prompt = s.prompt
            gen = self._engine.generate_video(
                media=s.media, prompt=prompt, decode=decode, model_cfg=self.model_cfg
            )
            gen.id = s.id
            gens.append(gen)
        return gens
