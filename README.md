# vlm-bench

**A thin, self-describing harness to benchmark open-source VLMs** (Qwen-VL, InternVL, etc.) across custom datasets — with pluggable backends (VLMEvalKit today, native/vLLM tomorrow), simple scoring, and reproducible run logs.

## Why vlm-bench?
- **Fast start:** use VLMEvalKit under the hood without inheriting its opinions everywhere.
- **Portable:** your dataset spec, prompts, scorers, and logs stay stable if you change backends.
- **Reproducible:** per-item JSONL traces + a bench lock file (env, git SHA, pip freeze).

## Install
```bash
git clone <your fork url> vlm-bench
cd vlm-bench
pip install -e .
# (optionally) add VLMEvalKit when you turn on that backend:
# pip install "vlmevalkit @ git+https://github.com/open-compass/VLMEvalKit.git"
```

## Quickstart
1) Put your dataset lines in `data/my_bench_dev.jsonl`:
```json
{"id":"vid_0001","task_type":"video_qa","media":"~/data/clips/clip1.mp4","prompt":"What color is the traffic light? Answer one word.","targets":["green"],"metadata":{"tags":["traffic-light"]}}
```

2) Tweak configs in `vlm_bench/configs/` (model, dataset, run).

3) Run:
```bash
vlm-bench run   --system vlm_bench/configs/system.yaml   --model  vlm_bench/configs/models/qwen_vl.yaml   --dataset vlm_bench/configs/datasets/my_bench.yaml   --run    vlm_bench/configs/runs/qwen2p5vl_7b.yaml
```

Artifacts:
- `vlm_bench/reports/<report>.jsonl` — per-item inputs, outputs, scores, timings  
- `vlm_bench/reports/<report>.parquet` — table form  
- `vlm_bench/reports/<report>.benchlock.json` — env snapshot

## Add a dataset
- Edit `vlm_bench/configs/datasets/my_bench.yaml` to point to your JSONL.
- Implement a new adapter under `registry/dataset_adapters/` if you need custom loading.

## Add a model family
- Add an adapter in `registry/model_adapters/` to normalize prompts, vision args, and decoding.
- Point a config at it in `configs/models/`.

## Backends
- **VLMEvalKit backend** (default): easy wins + existing wrappers.
- **Native backend**: use `runners/native_backend.py` when you want transformers/vLLM control.

## Scoring
Built-ins: `exact`, `numeric_tol`, `regex_extract`.  
For subjective tasks, swap in `llm_judge_stub.py` with your cached judge.

## License
MIT (see `LICENSE`).
