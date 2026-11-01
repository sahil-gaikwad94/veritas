"""Phase 3: CDCL jumpback, Tabu trie, domain cuts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClauseCheckpoint:
    clause_id: Any
    var_id: Any = None
    committed_val: float = 0.0
    start_token_idx: int = 0
    end_token_idx: int = 0
    mean_shannon_entropy: float = 0.0
    token_sequence: list[Any] = field(default_factory=list)
    conflict_count: int = 0


@dataclass
class ConflictBranchRecord:
    prefix_tokens: list[Any] = field(default_factory=list)
    rejected_clause_tokens: list[Any] = field(default_factory=list)
    unsat_core_ids: list[Any] = field(default_factory=list)
    violation_margin: float = 0.0


class TabuPrefixTrie:
    def __init__(self):
        self.entries: dict[int, set] = {}

    def add(self, start_idx: int, seq: list[Any]) -> None:
        self.entries.setdefault(start_idx, set()).add(tuple(seq))

    def contains(self, start_idx: int, seq: list[Any]) -> bool:
        return tuple(seq) in self.entries.get(start_idx, set())


class CdclJumpbackEngine:
    def __init__(self, lambda_: float = 0.5, gamma: float = 0.1):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.checkpoints: list[ClauseCheckpoint] = []
        self.tabu = TabuPrefixTrie()
        self.domain_cuts: list[dict[str, Any]] = []
        self.records: list[ConflictBranchRecord] = []

    def push_checkpoint(self, cp: ClauseCheckpoint) -> None:
        self.checkpoints.append(cp)

    def culprit_score(self, cp: ClauseCheckpoint, current_clause: Any) -> float:
        try:
            cur = float(current_clause)
        except (TypeError, ValueError):
            cur = float(len(self.checkpoints))
        # current_clause - clause_id; if clause_id not numeric, use index
        try:
            diff = cur - float(cp.clause_id)
        except (TypeError, ValueError):
            diff = float(len(self.checkpoints) - self.checkpoints.index(cp) - 1)
        return cp.mean_shannon_entropy * math.exp(-self.lambda_ * diff) + self.gamma * cp.conflict_count

    def select_culprit(self, unsat_core: list[Any]) -> ClauseCheckpoint | None:
        candidates = [cp for cp in self.checkpoints if cp.clause_id in unsat_core]
        if not candidates:
            # fallback: last checkpoint
            return self.checkpoints[-1] if self.checkpoints else None
        return max(candidates, key=lambda cp: self.culprit_score(cp, unsat_core[-1] if unsat_core else len(self.checkpoints)))

    def jumpback(self, unsat_core: list[Any]) -> dict[str, Any]:
        culprit = self.select_culprit(unsat_core)
        if culprit is None:
            return {"target_clause": None, "rewind_to": None}
        # record tabu
        self.tabu.add(culprit.start_token_idx, culprit.token_sequence)
        # domain cut
        cut = {
            "var_id": culprit.var_id,
            "committed_val": culprit.committed_val,
            "depth": culprit.clause_id,
        }
        self.domain_cuts.append(cut)
        # record conflict branch
        rec = ConflictBranchRecord(
            prefix_tokens=list(culprit.token_sequence),
            rejected_clause_tokens=list(culprit.token_sequence),
            unsat_core_ids=list(unsat_core),
            violation_margin=0.0,
        )
        self.records.append(rec)
        return {
            "target_clause": culprit.clause_id,
            "rewind_to": culprit.start_token_idx - 1 if culprit.start_token_idx > 0 else 0,
            "culprit": culprit,
            "tabu_added": True,
            "domain_cut": cut,
        }

    def apply_domain_cut(self, culprit: ClauseCheckpoint, direction: str = "upper") -> dict[str, Any]:
        if direction == "upper":
            return {"var_id": culprit.var_id, "new_upper": culprit.committed_val}
        return {"var_id": culprit.var_id, "new_lower": culprit.committed_val}

    def export_records(self) -> list[ConflictBranchRecord]:
        return list(self.records)


# aliases
CdclEngine = CdclJumpbackEngine
JumpbackEngine = CdclJumpbackEngine
DomainCutEngine = CdclJumpbackEngine
TabuTrie = TabuPrefixTrie
ConflictRecorder = CdclJumpbackEngine
UnsatCoreAttributor = CdclJumpbackEngine
EntropyWeightedJumpback = CdclJumpbackEngine
CDCL = CdclJumpbackEngine
