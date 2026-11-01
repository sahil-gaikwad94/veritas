"""veritas.rl collector (Phase 4)."""

from __future__ import annotations

from typing import Any, List, Dict
from dataclasses import dataclass, field


@dataclass
class ConflictBranchRecord:
    prefix_tokens: List[Any] = field(default_factory=list)
    rejected_clause_tokens: List[Any] = field(default_factory=list)
    unsat_core_ids: List[Any] = field(default_factory=list)
    violation_margin: float = 0.0


class VeritasRLCollector:
    def __init__(self):
        self.records: List[ConflictBranchRecord] = []

    def record_conflict_branch(self, state, clause, core_nodes) -> None:
        self.records.append(ConflictBranchRecord(unsat_core_ids=list(core_nodes or [])))

    def export_step_dpo(self, path: str = "traces/") -> Dict[str, Any]:
        return {"path": path, "num_records": len(self.records), "format": "dpo"}

    def export_grpo(self, path: str = "traces/") -> Dict[str, Any]:
        return {"path": path, "num_records": len(self.records), "format": "grpo"}


def export_step_dpo(path: str = "traces/"):
    return VeritasRLCollector().export_step_dpo(path)
