from typing import List, Any, Dict
import time
from vlm_bench.registry.model_adapters.base import ModelAdapter
from vlm_bench.registry.dataset_adapters.base import Sample, Generation
from vlm_bench.utils.video import select_frames

class VLMEvalKitAdapter(ModelAdapter):
    family = "vlmevalkit"

    def __init__(self, model_cfg: Dict[str, Any] | None = None):
        # model_cfg is kept for compatibility; CLI should pass the model cfg here.
        self.model_cfg = model_cfg or {}
        self._vlmeval = None

    def load(self):
        # Lazy import; install VLMEvalKit to enable
        try:
            import importlib
            # TODO: import the actual entry points you will use from VLMEvalKit
            self._vlmeval = importlib.import_module("vlmevalkit")
        except Exception:
            self._vlmeval = None  # will error on generate if used without install

    # Existing image/text path (kept for compatibility with other adapters)
    def generate(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Generation]:
        if self._vlmeval is None:
            raise RuntimeError("VLMEvalKit not installed. Install it or switch runner to 'native'.")
        gens: List[Generation] = []
        for s in batch:
            text = f"[VLMEvalKit placeholder output for {s.id}]"
            gens.append(Generation(id=s.id, text=text))
        return gens

    # New: video generation
    def generate_video(self, media: List[str], prompt: str, decode: Dict[str, Any], model_cfg: Dict[str, Any]) -> Generation:
        if self._vlmeval is None:
            raise RuntimeError("VLMEvalKit not installed. Install it or switch runner to 'native'.")
        t0 = time.time()

        # 1) Prepare frames if media is a single .mp4; otherwise assume frames provided
        video_cfg = (model_cfg.get("video") or {})
        if len(media) == 1 and media[0].lower().endswith(".mp4"):
            frames, vmeta = select_frames(
                media[0],
                max_frames=video_cfg.get("max_frames", 32),
                strategy=video_cfg.get("strategy", "uniform"),
                fps=video_cfg.get("fps"),
                stride=video_cfg.get("stride"),
            )
        else:
            frames, vmeta = media, {
                "selected_indices": list(range(len(media))),
                "num_frames_total": len(media),
                "strategy": "provided",
                "fps_used": None,
            }

        # 2) TODO: invoke real VLMEvalKit InternVL3 wrapper here.
        #    Example (pseudo):
        # model_name = model_cfg.get("hf_repo") or "OpenGVLab/InternVL3-38B"
        # wrapper = self._vlmeval.load_model(model_name, backend="internvl3")
        # text = wrapper.generate_video(frames, prompt, **decode)
        text = "[VLMEvalKit internvl3 video placeholder]"

        gen = Generation(
            id="",
            text=text,
            tokens=None,
            latency_ms=(time.time() - t0) * 1000,
            truncated=bool(len(text) >= decode.get("max_new_tokens", 1024) - 5)
        )
        return gen
