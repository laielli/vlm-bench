import argparse, time
from pathlib import Path
from vlm_bench.utils.logging import get_logger
from vlm_bench.utils.io import write_jsonl, write_parquet
from vlm_bench.utils.seed import seed_everything
from vlm_bench.benchlock import write_bench_lock

# adapters & runners
from vlm_bench.registry.dataset_adapters.my_bench import MyBenchAdapter
from vlm_bench.registry.model_adapters.qwen_vl import QwenVLAdapter
from vlm_bench.registry.model_adapters.internvl import InternVLAdapter
from vlm_bench.runners.base import Runner
from vlm_bench.runners.vlmeval_backend import VLMEvalKitAdapter
from vlm_bench.runners.native_backend import NativeAdapter
from vlm_bench.scorers.exact import ExactMatch
from vlm_bench.scorers.numeric_tol import NumericTolerance
from vlm_bench.scorers.regex_extract import RegexExtract

ADAPTERS = {
    "dataset": {"my_bench": MyBenchAdapter},
    "model": {"qwen-vl": QwenVLAdapter, "internvl": InternVLAdapter},
    "engine": {"vlmeval": VLMEvalKitAdapter, "native": NativeAdapter},
    "scorer": {"exact": ExactMatch, "numeric_tol": NumericTolerance, "regex_extract": RegexExtract},
}

def main():
    ap = argparse.ArgumentParser(prog="vlm-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a benchmark")
    run.add_argument("--system", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--run", required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        import yaml
        sys_cfg = yaml.safe_load(open(args.system))
        mdl_cfg = yaml.safe_load(open(args.model))
        dset_cfg = yaml.safe_load(open(args.dataset))
        run_cfg  = yaml.safe_load(open(args.run))["run"]

        log = get_logger()
        seed_everything(run_cfg.get("seed", 1234))

        out_dir = Path(sys_cfg["paths"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

        engine_cls = ADAPTERS["engine"][sys_cfg["runner"]]
        engine = engine_cls(model_cfg={}); engine.load()

        mdl_cfg["__shared_engine"] = engine
        model_cls = ADAPTERS["model"][mdl_cfg["family"]]
        model = model_cls(mdl_cfg); model.load()

        ds_cls = ADAPTERS["dataset"][dset_cfg["adapter"]]
        dataset = ds_cls(dset_cfg)

        sc_name = sys_cfg["scorer"]["name"]
        scorer = ADAPTERS["scorer"][sc_name](**sys_cfg["scorer"].get("params", {}))

        runner = Runner(model=model, scorer=scorer, batch_size=run_cfg.get("batch_size",1))
        decode = model.default_decode(); decode.update(mdl_cfg.get("decode", {}))

        log.info("Starting run...")
        t0 = time.time()
        results = runner.run(samples=dataset, decode=decode)
        log.info(f"Finished {len(results)} items in {time.time()-t0:.2f}s")

        name = run_cfg.get("report_name", "report")
        write_jsonl(out_dir / f"{name}.jsonl", results)
        try:
            write_parquet(out_dir / f"{name}.parquet", results)
        except Exception as e:
            log.warning(f"Parquet save failed: {e}")
        write_bench_lock(out_dir / f"{name}.benchlock.json")
