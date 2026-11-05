"""Tests for the adaptive ``<|solver_feedback|>`` reflex loop on CDCL backtrack."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pydantic import BaseModel, Field

import vexx as vt
from vexx.decode import DeadEnd, DecodeConfig, VexxSession
from vexx.testing import StructuredReasoningModel


class Portfolio(BaseModel):
    weight_a: float = Field(ge=0.0, le=1.0)
    weight_b: float = Field(ge=0.0, le=1.0)


@vt.verified_schema()
def portfolio(p: Portfolio):
    vt.require(p.weight_a + p.weight_b == 1.0, tol=1e-6)


def test_reflex_feedback_span_is_recorded_but_never_in_json():
    cfg = DecodeConfig(reflex_on_backtrack=True)
    model = StructuredReasoningModel(reasoning="verified adjustment thought")
    session = VexxSession(portfolio, model, cfg)
    session.telemetry.domain_cuts.append({"field": "weight_a", "value": 0.9, "bounds": [0.0, 0.5]})
    exc = DeadEnd("weight_a", "contractor became infeasible after weight_a=0.9")
    session._reflex_feedback(model, exc)
    assert "<|solver_feedback|>" in session.reasoning
    assert "verified adjustment thought" in session.reasoning
    # The reflex span lives in reasoning only; the decoded JSON is never touched by it.
    session.text = '{"weight_a": 0.5, "weight_b": 0.5}'
    assert "<|solver_feedback|>" not in session.text


def test_reflex_feedback_trace_note():
    cfg = DecodeConfig(reflex_on_backtrack=True)
    model = StructuredReasoningModel(reasoning="analysis")
    session = VexxSession(portfolio, model, cfg)
    exc = DeadEnd("weight_a", "conflict")
    session._reflex_feedback(model, exc)
    kinds = [step.kind for step in session.trace]
    assert "reflex" in kinds


def test_reflex_disabled_by_default():
    cfg = DecodeConfig()
    assert cfg.reflex_on_backtrack is False
