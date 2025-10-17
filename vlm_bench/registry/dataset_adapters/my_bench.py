import json, os
from typing import Iterator
from vlm_bench.registry.dataset_adapters.base import DatasetAdapter, Sample

class MyBenchAdapter(DatasetAdapter):
    def __iter__(self) -> Iterator[Sample]:
        items_path = self.cfg["items_path"]
        with open(items_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                media = ex["media"] if isinstance(ex["media"], list) else [ex["media"]]
                yield Sample(
                    id=str(ex["id"]),
                    task_type=ex.get("task_type","vqa"),
                    media=[os.path.expanduser(p) for p in media],
                    prompt=ex["prompt"],
                    targets=ex.get("targets", []),
                    metadata=ex.get("metadata", {}),
                )
