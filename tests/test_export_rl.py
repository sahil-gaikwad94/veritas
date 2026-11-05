"""Tests for ``vexx export-rl`` and decoder conflict-trace capture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vexx.cli.main import main

TRACES = [
    {
        "kind": "conflict",
        "prompt": "weights sum to one",
        "chosen": '{"w": 0.5, "x": 0.5}',
        "rejected": '{"w": 0.9, "x": 0.4}',
        "unsat_core_ids": [0, 1],
        "violation_margin": 0.4,
        "metadata": {"spec_digest": "abc123", "culprit_field": "w"},
        "prefix_tokens": ["weights"],
        "rejected_clause_tokens": ["0.9"],
    },
    {
        "kind": "conflict",
        "prompt": "weights sum to one",
        "chosen": "",
        "rejected": '{"w": 0.9, "x": 0.1}',
        "unsat_core_ids": [1],
        "violation_margin": 0.1,
        "metadata": {},
        "prefix_tokens": ["weights"],
        "rejected_clause_tokens": ["0.1"],
    },
    {
        "kind": "step",
        "prompt": "weights sum to one",
        "completion": '{"w": 0.5, "x": 0.5}',
        "rewards": [1.0, 0.5, 0.1],
        "metadata": {"episode": 1},
    },
]


def _write_trace(tmp_path: Path) -> Path:
    target = tmp_path / "traces.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for record in TRACES:
            handle.write(json.dumps(record) + "\n")
    return target


def test_export_dpo(tmp_path):
    trace = _write_trace(tmp_path)
    out = tmp_path / "dpo.jsonl"
    code = main(["export-rl", str(trace), "--format", "dpo", "-o", str(out)])
    assert code == 0
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1  # the record without `chosen` is dropped
    assert lines[0]["prompt"] == "weights sum to one"
    assert lines[0]["chosen"] != lines[0]["rejected"]
    assert lines[0]["metadata"]["unsat_core"] == [0, 1]
    manifest = out.with_suffix(out.suffix + ".manifest.json")
    assert manifest.exists()
    assert json.loads(manifest.read_text())["num_records"] == 1


def test_export_grpo(tmp_path):
    trace = _write_trace(tmp_path)
    out = tmp_path / "grpo.jsonl"
    code = main(["export-rl", str(trace), "--format", "grpo", "-o", str(out)])
    assert code == 0
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) >= 1
    total = sum(line["reward"] for line in lines if "reward" in line)
    assert total > 0


def test_export_rejects_malformed_lines(tmp_path):
    trace = tmp_path / "bad.jsonl"
    trace.write_text("not json\n{\"kind\": \"step\", \"prompt\": \"p\", \"completion\": \"c\", \"rewards\": [1.0]}\n")
    out = tmp_path / "out.jsonl"
    code = main(["export-rl", str(trace), "--format", "grpo", "-o", str(out)])
    assert code == 0
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
