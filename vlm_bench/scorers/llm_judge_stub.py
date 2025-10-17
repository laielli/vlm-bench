from vlm_bench.scorers.base import Scorer
from vlm_bench.registry.dataset_adapters.base import Sample, Generation

class LLMJudgeStub(Scorer):
    name = "llm_judge"
    def score(self, sample: Sample, gen: Generation):
        return {"metric": "judge", "value": None, "reason": "not_implemented"}
