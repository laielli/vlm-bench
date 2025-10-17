from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Iterator

TaskType = Literal["vqa","caption","classification","ocr","math","video_qa"]

@dataclass
class Sample:
    id: str
    task_type: TaskType
    media: List[str]
    prompt: str
    targets: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class Generation:
    id: str
    text: str
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    truncated: Optional[bool] = None

class DatasetAdapter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
    def __iter__(self) -> Iterator[Sample]:
        raise NotImplementedError
