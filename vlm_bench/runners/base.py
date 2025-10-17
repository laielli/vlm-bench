from typing import Any, Dict, Iterable, List
from vlm_bench.registry.dataset_adapters.base import Sample, Generation
from vlm_bench.registry.model_adapters.base import ModelAdapter

class Runner:
    def __init__(self, model: ModelAdapter, scorer, batch_size: int = 1):
        self.model = model
        self.scorer = scorer
        self.batch_size = batch_size

    def run(self, samples: Iterable[Sample], decode: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        batch: List[Sample] = []
        for s in samples:
            batch.append(s)
            if len(batch) == self.batch_size:
                out += self._run_batch(batch, decode)
                batch = []
        if batch:
            out += self._run_batch(batch, decode)
        return out

    def _run_batch(self, batch: List[Sample], decode: Dict[str, Any]) -> List[Dict[str, Any]]:
        gens: List[Generation] = self.model.generate(batch, decode)
        results = []
        for s, g in zip(batch, gens):
            score = self.scorer.score(s, g)
            results.append({
                "id": s.id, "task_type": s.task_type,
                "prompt": s.prompt, "targets": s.targets,
                "media": s.media, "generation": g.__dict__, "score": score
            })
        return results
