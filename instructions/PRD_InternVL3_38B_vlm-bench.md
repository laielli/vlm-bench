# PRD: InternVL3-38B Support in `vlm-bench` (Video QA)

## 1) Objective
Enable running **InternVL3-38B** on **video-QA** tasks inside `vlm-bench`, using:
- **Backend A:** VLMEvalKit (if installed)
- **Backend B:** Native Transformers (CPU/GPU; with optional 8-bit/4-bit quant)

Deliver consistent per-item logs, scoring, and reproducibility—reusing `vlm-bench` abstractions.

## 2) In Scope
- Model adapter: `InternVL3Adapter` (family: `"internvl3"`)
- Two runtimes:
  - `VLMEvalKitAdapter.generate_video(...)`
  - `NativeAdapter.generate_video(...)`
- **Video ingestion**: configurable frame sampling policy (uniform, stride, max_frames)
- Decode defaults for long-form answers (avoid truncation)
- Configs + CLI wiring
- CI smoke for non-GPU path (mock/echo); local GPU README instructions

## 3) Out of Scope
- vLLM runtime
- Audio inputs
- Training/finetuning
- LLM-as-judge implementation

## 4) Success Criteria (Acceptance)
- ✅ `vlm-bench run ...` completes for a small video-QA JSONL using both backends (when deps present).
- ✅ Per-item JSONL includes: selected frame indices, decode config, latency, and non-truncation flag.
- ✅ Scores computed via existing scorers (e.g., `exact`), no crashes.
- ✅ Deterministic results under fixed seed + deterministic sampler.
- ✅ Clear error if VLMEvalKit or model weights not installed.

## 5) Functional Requirements
### 5.1 Model Adapter
- New file: `vlm_bench/registry/model_adapters/internvl3.py`
- `family = "internvl3"`
- Methods:
  - `load(self)` — set `self._engine` from `__shared_engine`; capture vision/video args.
  - `default_decode(self)` — `{temperature: 0.0, max_new_tokens: 1024, stop: []}`
  - `generate(self, batch, decode)` — for each `Sample`:
    - Accept **mp4 path** or a **list of frame paths**.
    - Build prompt (no chat template unless required by model).
    - Call `self._engine.generate_video(media, prompt, decode, model_cfg=self.model_cfg)`.
    - Return `Generation(id, text, tokens?, latency_ms?, truncated?)`.

### 5.2 Engines
#### 5.2.1 VLMEvalKitAdapter
- Add `generate_video(...)` that uses the shared frame sampler for mp4 inputs (unless VLMEvalKit can take video directly).
- Resolve model name from `model_cfg["hf_repo"]` or `["model_name"]`.
- Return a `Generation` object and include `latency_ms` and `truncated`.

#### 5.2.2 NativeAdapter
- `load()` uses `transformers` to fetch model + processor from Hugging Face `hf_repo`.
- Support quantization via `bitsandbytes` (`quantization: int8|int4`).
- `generate_video(...)` samples frames, preprocesses via processor, then calls `.generate`.
- Return `Generation` with timing and truncation detection.

### 5.3 Video Frame Sampler (Shared)
- New module: `vlm_bench/utils/video.py`
- `select_frames(video_path, max_frames=32, strategy="uniform", fps=None, stride=None) -> (frames, meta)`
- Prefer `decord` (fast); fallback to `opencv`. Deterministic under fixed seed.
- `meta` includes: `selected_indices`, `num_frames_total`, `strategy`, `fps_used`.

### 5.4 Config Schema
`vlm_bench/configs/models/internvl3_38b.yaml`
```yaml
family: internvl3
engine: vlmeval              # alt: native
hf_repo: OpenGVLab/InternVL3-38B   # confirm exact repo id when wiring
revision: main               # optional
quantization: none           # none | int8 | int4 (native backend only)
vision_args:
  short_side: 448
video:
  max_frames: 32
  strategy: uniform          # uniform | stride
  fps: null
  stride: null
decode:
  temperature: 0.0
  max_new_tokens: 1024
  stop: []
```

**Precedence:** dataset.video overrides model.video, which overrides hardcoded defaults.

### 5.5 Logging
For each item, include:
- `video.selected_indices`, `video.num_frames_total`, `video.strategy`, `video.fps_used`
- `decode.max_new_tokens`, `latency_ms`, `truncated`

### 5.6 Errors
- Missing VLMEvalKit → `RuntimeError("VLMEvalKit not installed...")`
- Missing HF weights → `RuntimeError("Hugging Face model not available: <repo>")`
- OOM → suggest `quantization: int8` and/or fewer frames

## 6) Non-Functional
- Reproducibility: deterministic frame selection and `temperature: 0` by default.
- Performance: default to `max_frames=32`; allow tuning.
- Compatibility: Python 3.10+, run without VLMEvalKit when `engine: native`.

## 7) File Changes (Checklist)
**New**
- `vlm_bench/registry/model_adapters/internvl3.py`
- `vlm_bench/utils/video.py`
- `vlm_bench/configs/models/internvl3_38b.yaml`

**Modified**
- `vlm_bench/runners/vlmeval_backend.py` — implement `generate_video(...)`
- `vlm_bench/runners/native_backend.py` — implement `load(...)` and `generate_video(...)`
- (Optional) `vlm_bench/cli.py` — ensure engine receives `model_cfg` in constructor

## 8) Developer Tasks
**P0**
- [ ] Implement the three code files in this package (replace TODOs).
- [ ] Wire the correct VLMEvalKit InternVL3 entry point.
- [ ] Wire the correct `transformers` classes for InternVL3 (model + processor).
- [ ] Add `configs/models/internvl3_38b.yaml` to repo.

**P1**
- [ ] Add sampler tests and an adapter smoke test.
- [ ] Add CI step with mock engine returning fixed text; upload artifacts.

**P2**
- [ ] README section: GPU + quant tips and usage examples.
