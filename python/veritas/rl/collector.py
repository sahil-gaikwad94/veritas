"""Step-level RL data collection from constraint conflicts and verified trajectories.

Every CDCL jumpback is a *process-level* preference pair: a rejected assignment and the
repaired assignment that satisfied the constraints.  This collector turns those events
into datasets that TRL-style trainers can consume directly:

* **Step-DPO** — ``{prompt, chosen, rejected, metadata}`` JSONL.
* **Process-GRPO** — dense per-step rewards derived from conflict distance and backtrack
  count, exported as ``{prompt, completion, rewards, reward}`` JSONL.

Exports are written atomically with a manifest (counts + SHA-256) so training runs are
reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class ConflictBranchRecord:
    prefix_tokens: List[Any] = field(default_factory=list)
    rejected_clause_tokens: List[Any] = field(default_factory=list)
    unsat_core_ids: List[Any] = field(default_factory=list)
    violation_margin: float = 0.0
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prefix_tokens": list(self.prefix_tokens),
            "rejected_clause_tokens": list(self.rejected_clause_tokens),
            "unsat_core_ids": list(self.unsat_core_ids),
            "violation_margin": float(self.violation_margin),
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": dict(self.metadata),
        }


@dataclass
class StepRecord:
    prompt: str
    completion: str
    rewards: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return float(sum(self.rewards)) if self.rewards else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "rewards": [round(r, 6) for r in self.rewards],
            "reward": round(self.reward, 6),
            "metadata": dict(self.metadata),
        }


class VeritasRLCollector:
    def __init__(self):
        self.records: List[ConflictBranchRecord] = []
        self.steps: List[StepRecord] = []

    # -- recording -------------------------------------------------------
    def record_conflict_branch(
        self,
        state: Any = None,
        clause: Any = None,
        core_nodes: Optional[Sequence[Any]] = None,
        *,
        record: Optional[ConflictBranchRecord] = None,
        prompt: str = "",
        rejected: str = "",
        margin: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConflictBranchRecord:
        if record is None:
            record = ConflictBranchRecord(
                unsat_core_ids=list(core_nodes or []),
                prompt=prompt,
                rejected=rejected or (repr(list(core_nodes)) if core_nodes else ""),
                violation_margin=float(margin),
                metadata=dict(metadata or {}),
            )
        else:
            if core_nodes:
                record.unsat_core_ids = list(core_nodes)
            if prompt:
                record.prompt = prompt
            if rejected:
                record.rejected = rejected
            record.violation_margin = max(record.violation_margin, float(margin))
        self.records.append(record)
        return record

    def record_repair(self, record: ConflictBranchRecord, chosen: str) -> ConflictBranchRecord:
        record.chosen = chosen
        return record

    def add_preference(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        *,
        unsat_core: Optional[Sequence[Any]] = None,
        margin: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConflictBranchRecord:
        record = ConflictBranchRecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            unsat_core_ids=list(unsat_core or []),
            violation_margin=float(margin),
            metadata=dict(metadata or {}),
        )
        self.records.append(record)
        return record

    def record_step(self, step: StepRecord) -> StepRecord:
        self.steps.append(step)
        return step

    # -- rewards ---------------------------------------------------------
    @staticmethod
    def process_reward(
        *,
        verified: bool,
        backtracks: int = 0,
        masked_tokens: int = 0,
        jump_forward_tokens: int = 0,
        final_margin: float = 0.0,
        max_backtracks: int = 8,
    ) -> List[float]:
        """Dense process rewards: verified completion + efficiency shaping."""

        terminal = 1.0 if verified else -1.0
        efficiency = 1.0 - min(1.0, backtracks / max(1, max_backtracks))
        mechanics = 1.0 if jump_forward_tokens >= 0 else 0.0
        return [
            terminal,
            0.25 * efficiency,
            0.10 * mechanics,
            -0.05 * max(0.0, final_margin),
        ]

    # -- exports ---------------------------------------------------------
    def export_step_dpo(self, path: str = "traces/") -> Dict[str, Any]:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "step_dpo.jsonl"
        rows = []
        for record in self.records:
            rows.append(
                {
                    "prompt": record.prompt,
                    "chosen": record.chosen,
                    "rejected": record.rejected,
                    "metadata": {
                        "unsat_core": record.unsat_core_ids,
                        "violation_margin": record.violation_margin,
                        **record.metadata,
                    },
                }
            )
        digest = _write_jsonl(target, rows)
        manifest = self._manifest(directory, {"step_dpo.jsonl": digest, "rows": len(rows)})
        return {
            "path": str(directory),
            "file": str(target),
            "num_records": len(rows),
            "total_conflicts": len(self.records),
            "format": "dpo",
            "sha256": digest,
            "manifest": manifest,
        }

    def export_grpo(self, path: str = "traces/") -> Dict[str, Any]:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "process_grpo.jsonl"
        rows = [step.as_dict() for step in self.steps]
        digest = _write_jsonl(target, rows)
        manifest = self._manifest(directory, {"process_grpo.jsonl": digest, "rows": len(rows)})
        return {
            "path": str(directory),
            "file": str(target),
            "num_records": len(rows),
            "format": "grpo",
            "sha256": digest,
            "manifest": manifest,
        }

    def to_hf_dataset(self, kind: str = "dpo") -> Dict[str, Any]:
        """Return a datasets-compatible payload without importing ``datasets``."""

        if kind == "grpo":
            return {"format": "grpo", "rows": [step.as_dict() for step in self.steps]}
        return {
            "format": "dpo",
            "rows": [
                {
                    "prompt": record.prompt,
                    "chosen": record.chosen,
                    "rejected": record.rejected,
                    "metadata": record.as_dict(),
                }
                for record in self.records
                if record.chosen or record.rejected
            ],
        }

    def _manifest(self, directory: Path, payload: Dict[str, Any]) -> str:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "conflicts": len(self.records),
            "steps": len(self.steps),
            **payload,
        }
        target = directory / "manifest.json"
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return str(target)

    def clear(self) -> None:
        self.records.clear()
        self.steps.clear()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, default=str)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            count += 1
    return digest.hexdigest()


def export_step_dpo(path: str = "traces/") -> Dict[str, Any]:
    return VeritasRLCollector().export_step_dpo(path)


__all__ = [
    "ConflictBranchRecord",
    "StepRecord",
    "VeritasRLCollector",
    "export_step_dpo",
]
