from typing import Dict
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class Scorer:
    name = "base"
    def score(self, sample: Sample, gen: Generation) -> Dict:
        raise NotImplementedError
