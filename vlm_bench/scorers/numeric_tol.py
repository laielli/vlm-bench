import math, re
from typing import Dict
from vlm_bench.scorers.base import Scorer
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

def _extract_number(s: str):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

class NumericTolerance(Scorer):
    name = "numeric_tol"
    def __init__(self, atol: float = 1e-3, rtol: float = 0.0):
        self.atol, self.rtol = atol, rtol
    def score(self, sample: Sample, gen: Generation) -> Dict:
        pred = _extract_number(gen.text)
        golds = [_extract_number(t) for t in sample.targets]
        ok = False
        if pred is not None:
            for g in golds:
                if g is None:
                    continue
                if math.isclose(pred, g, rel_tol=self.rtol, abs_tol=self.atol):
                    ok = True; break
        return {"metric": "acc_num", "value": 1.0 if ok else 0.0, "pred_num": pred}
