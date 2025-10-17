import re
from typing import Dict
from vlm_bench.scorers.base import Scorer
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class RegexExtract(Scorer):
    name = "regex_extract"
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern, re.I | re.S)
    def score(self, sample: Sample, gen: Generation) -> Dict:
        m = self.pattern.search(gen.text)
        extracted = m.group(1).strip() if m else gen.text.strip()
        correct = extracted.lower() in [t.lower() for t in sample.targets]
        return {"metric": "acc_rex", "value": 1.0 if correct else 0.0, "extracted": extracted}
