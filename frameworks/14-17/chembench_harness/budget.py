from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised before an action that would exceed a declared hard budget."""


@dataclass
class BudgetManager:
    limits: dict[str, int]
    used: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.limits.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"budget {name!r} must be a non-negative integer")
        unknown = set(self.used) - set(self.limits)
        if unknown:
            raise ValueError(f"usage contains undeclared budgets: {sorted(unknown)}")
        for name, value in self.used.items():
            if not isinstance(value, int) or value < 0 or value > self.limits[name]:
                raise ValueError(f"invalid usage for budget {name!r}: {value}")

    def remaining(self, name: str) -> int:
        if name not in self.limits:
            raise KeyError(f"undeclared budget: {name}")
        return self.limits[name] - self.used.get(name, 0)

    def consume(self, name: str, amount: int = 1) -> None:
        if not isinstance(amount, int) or amount <= 0:
            raise ValueError("budget consumption must be a positive integer")
        if self.remaining(name) < amount:
            raise BudgetExceeded(
                f"budget {name!r} exhausted: requested {amount}, "
                f"remaining {self.remaining(name)}"
            )
        self.used[name] = self.used.get(name, 0) + amount

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "limit": limit,
                "used": self.used.get(name, 0),
                "remaining": limit - self.used.get(name, 0),
            }
            for name, limit in sorted(self.limits.items())
        }
