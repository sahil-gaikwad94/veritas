"""veritas.code_synth cartridge."""

from __future__ import annotations

from typing import Any, Set, List


def scope_safe(identifiers: Set[str], used: Set[str]) -> bool:
    return used.issubset(identifiers)


def no_division_by_zero(divisors: List[float]) -> bool:
    return all(d != 0 for d in divisors)


def array_bounds_ok(index: int, size: int) -> bool:
    return 0 <= index < size
