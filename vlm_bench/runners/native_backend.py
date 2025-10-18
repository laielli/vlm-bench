import time
from typing import List, Any, Dict
from vlm_bench.registry.dataset_adapters.base import Generation
from vlm_bench.utils.video import select_frames

class NativeAdapter:
    """Lightweight native engine using transformers.
    TODOs:
      - Replace Auto* classes with InternVL3-compatible classes per model card.
      - Implement proper multimodal processing (video frames + prompt).
    """
    def __init__(self, model_cfg: Dict[str, Any] | None = None):
        self.model_cfg = model_cfg or {}
        self.model = None
        self.processor = None
        self.device = "cpu"

    def load(self):
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"

        repo = self.model_cfg.get("hf_repo", None)
        if not repo:
            # Allow init without weights (e.g., CI smoke with MOCK_ENGINE), but warn.
            return

        # NOTE: These are placeholders; replace with actual classes per InternVL3 repo.
        from transformers import AutoModelForCausalLM, AutoProcessor  # TODO: adjust for multimodal class
        quant = (self.model_cfg.get("quantization") or "none").lower()
        kwargs = {}
        if quant in ("int8", "int4"):
            # bitsandbytes loads
            kwargs.update(dict(
                device_map="auto",
                load_in_8bit=(quant == "int8"),
                load_in_4bit=(quant == "int4"),
            ))
        self.model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        self.processor = AutoProcessor.from_pretrained(repo)
        if self.device == "cuda":
            self.model.to("cuda")

    def generate_video(self, media: List[str], prompt: str, decode: Dict[str, Any], model_cfg: Dict[str, Any]) -> Generation:
        # MOCK/placeholder path for CPUs or CI: if no model present, return stub text
        t0 = time.time()
        if self.model is None or self.processor is None:
            text = "[native internvl3 video placeholder]"
            return Generation(id="", text=text, latency_ms=(time.time()-t0)*1000, truncated=False)

        # 1) Collect frames
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

        # 2) TODO: Proper multimodal encode for InternVL3. Pseudocode:
        # from transformers import GenerationConfig
        # inputs = self.processor(images=frames, text=prompt, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # gen_ids = self.model.generate(
        #     **inputs,
        #     max_new_tokens=decode.get("max_new_tokens", 1024),
        #     do_sample=decode.get("temperature", 0.0) > 0.0,
        #     temperature=decode.get("temperature", 0.0),
        # )
        # text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        text = "[native internvl3 video placeholder]"  # remove once real call is wired
        truncated = len(text) >= decode.get("max_new_tokens", 1024) - 5
        return Generation(id="", text=text, latency_ms=(time.time()-t0)*1000, truncated=truncated)
