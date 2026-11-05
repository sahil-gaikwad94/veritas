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
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConflictBranchRecord:
    prefix_tokens: list[Any] = field(default_factory=list)
    rejected_clause_tokens: list[Any] = field(default_factory=list)
    unsat_core_ids: list[Any] = field(default_factory=list)
    violation_margin: float = 0.0
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
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
    rewards: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return float(sum(self.rewards)) if self.rewards else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "rewards": [round(r, 6) for r in self.rewards],
            "reward": round(self.reward, 6),
            "metadata": dict(self.metadata),
        }


class VexxRLCollector:
    def __init__(self):
        self.records: list[ConflictBranchRecord] = []
        self.steps: list[StepRecord] = []

    # -- recording -------------------------------------------------------
    def record_conflict_branch(
        self,
        state: Any = None,
        clause: Any = None,
        core_nodes: Sequence[Any] | None = None,
        *,
        record: ConflictBranchRecord | None = None,
        prompt: str = "",
        rejected: str = "",
        margin: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
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
        unsat_core: Sequence[Any] | None = None,
        margin: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
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
    ) -> list[float]:
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
    def export_step_dpo(self, path: str = "traces/") -> dict[str, Any]:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "step_dpo.jsonl"
        rows = []
        for record in self.records:
            if not (record.prompt and record.chosen and record.rejected) or record.chosen == record.rejected:
                continue
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

    def export_grpo(self, path: str = "traces/") -> dict[str, Any]:
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

    def to_hf_dataset(self, kind: str = "dpo") -> dict[str, Any]:
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
                if record.prompt and record.chosen and record.rejected and record.chosen != record.rejected
            ],
        }

    def _manifest(self, directory: Path, payload: dict[str, Any]) -> str:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "conflicts": len(self.records),
            "steps": len(self.steps),
            **payload,
        }
        target = directory / "manifest.json"
        _atomic_write(target, json.dumps(manifest, indent=2, sort_keys=True))
        return str(target)

    def clear(self) -> None:
        self.records.clear()
        self.steps.clear()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    lines = []
    for row in rows:
        line = json.dumps(row, sort_keys=True, default=str)
        lines.append(line + "\n")
        digest.update((line + "\n").encode("utf-8"))
    _atomic_write(path, "".join(lines))
    return digest.hexdigest()


def export_step_dpo(path: str = "traces/") -> dict[str, Any]:
    return VexxRLCollector().export_step_dpo(path)


__all__ = [
    "ConflictBranchRecord",
    "StepRecord",
    "VexxRLCollector",
    "export_step_dpo",
]
