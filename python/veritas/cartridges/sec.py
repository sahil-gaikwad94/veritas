"""veritas.sec cartridge."""

from __future__ import annotations

from typing import Any, List, Set


class WorldState:
    def __init__(self):
        self.facts: Set[str] = set()
        self.history: List[Any] = []

    def assert_fact(self, fact: str) -> None:
        self.facts.add(fact)
        self.history.append(("assert", fact))

    def has(self, fact: str) -> bool:
        return fact in self.facts

    def before(self, a: str, b: str) -> bool:
        # true if a occurred before b in history
        ia = next((i for i, h in enumerate(self.history) if h == ("assert", a)), None)
        ib = next((i for i, h in enumerate(self.history) if h == ("assert", b)), None)
        if ia is None or ib is None:
            return False
        return ia < ib

    def always(self, condition: bool) -> bool:
        return bool(condition)

    def can_call(self, tool: str, preconditions: List[str]) -> bool:
        return all(self.has(p) for p in preconditions)
