from dataclasses import dataclass
from typing import Optional


@dataclass
class MSASeq:
    id: int
    name: str
    seq: str
    ref: bool = False
    proba: Optional[float] = None
    logproba: Optional[float] = None
