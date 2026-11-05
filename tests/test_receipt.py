"""Receipt (KL distortion) and verification artifact tests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


import vexx as vt
from vexx.decode import _mask_kl
from vexx.testing import MockCharModel

SCHEMA_SOURCE = '''
import vexx as vt
from pydantic import BaseModel, Field

class Portfolio(BaseModel):
    weight_a: float = Field(ge=0.0, le=1.0)
    weight_b: float = Field(ge=0.0, le=1.0)

@vt.verified_schema()
def portfolio(p: Portfolio):
    vt.require(p.weight_a + p.weight_b == 1.0, tol=1e-6)
'''


def _schema_module(tmp_path: Path) -> Path:
    target = tmp_path / "portfolio.py"
    target.write_text(SCHEMA_SOURCE, encoding="utf-8")
    return target


def test_mask_kl_is_zero_for_unmasked_distributions():
    raw = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert _mask_kl(raw, ["a", "b", "c"]) == 0.0


def test_mask_kl_measures_lost_mass():
    raw = {"x": 10.0, "y": -100.0}
    # Allowing only y, almost all probability mass is discarded.
    distortion = _mask_kl(raw, ["y"])
    assert distortion > 10.0
    assert _mask_kl(raw, ["x", "y"]) == 0.0


def test_receipt_reports_content_and_distortion(tmp_path):
    schema_path = _schema_module(tmp_path)
    from vexx.cli.main import load_schemas

    schema = load_schemas(str(schema_path))[0]
    result = vt.generate(model=MockCharModel(), prompt="port", spec=schema)
    receipt = result.receipt()
    assert receipt["verified"] is True
    assert receipt["status"] == "PASS"
    assert len(receipt["sha256"]) == 64
    assert len(receipt["spec_digest"]) == 64
    assert "field_kl_divergence" in receipt
    assert receipt["field_kl_divergence"], "expected per-field KL distortion to be recorded"
    assert receipt["jump_forward_tokens"] >= 0
    digest = hashlib.sha256(result.raw.encode()).hexdigest()
    assert receipt["sha256"] == digest


def test_receipt_attempts_and_counters(tmp_path):
    schema_path = _schema_module(tmp_path)
    from vexx.cli.main import load_schemas

    schema = load_schemas(str(schema_path))[0]
    result = vt.generate(model=MockCharModel(), prompt="port", spec=schema)
    receipt = result.receipt()
    assert receipt["attempts"] >= 1
    assert receipt["backtracks"] >= 0
    assert receipt["elapsed_ms"] >= 0.0
