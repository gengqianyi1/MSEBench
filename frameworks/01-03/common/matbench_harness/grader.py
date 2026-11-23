from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


def normalize_reward(raw_score: float, max_raw_score: float, min_raw_score: float = 0) -> float:
    values = (raw_score, max_raw_score, min_raw_score)
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
        raise ValueError("score bounds must be finite numbers")
    if max_raw_score <= min_raw_score:
        raise ValueError("max_raw_score must be greater than min_raw_score")
    reward = (float(raw_score) - min_raw_score) / (max_raw_score - min_raw_score)
    return max(0.0, min(1.0, reward))


@dataclass(frozen=True)
class GraderResult:
    raw_score: float
    max_raw_score: float
    min_raw_score: float = 0.0
    status: str = "scored"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    schema_version: str = "chembench-grader-result/1.0"

    @property
    def reward(self) -> float:
        return normalize_reward(self.raw_score, self.max_raw_score, self.min_raw_score)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reward"] = self.reward
        return result
