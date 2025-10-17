import re
from typing import Dict
from vlm_bench.scorers.base import Scorer
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\-:%]", "", s)
    return s

class ExactMatch(Scorer):
    name = "exact"
    def score(self, sample: Sample, gen: Generation) -> Dict:
        pred = _normalize(gen.text)
        golds = [_normalize(t) for t in sample.targets]
        correct = pred in golds
        return {"metric": "acc", "value": 1.0 if correct else 0.0, "pred": gen.text}
