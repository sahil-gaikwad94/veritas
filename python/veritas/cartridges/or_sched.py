"""veritas.or_sched cartridge."""

from __future__ import annotations

from typing import List, Tuple


def no_overlap(intervals: List[Tuple[float, float]]) -> bool:
    sorted_iv = sorted(intervals)
    for i in range(1, len(sorted_iv)):
        if sorted_iv[i][0] < sorted_iv[i-1][1]:
            return False
    return True


def precedence_ok(starts: dict, durations: dict, preds: dict) -> bool:
    for task, pres in preds.items():
        for p in pres:
            if starts[task] < starts[p] + durations[p]:
                return False
    return True


def capacity_ok(loads: List[float], capacity: float) -> bool:
    return all(load <= capacity for load in loads)
