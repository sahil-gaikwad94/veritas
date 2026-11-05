"""Concrete instance verification, derivation and VIR serialization."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pydantic import BaseModel

import vexx as vt
from vexx.instance import verify_values
from vexx.serde import spec_digest, spec_from_dict, spec_to_dict


class Linear(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def linear(p: Linear):
    vt.require(p.a >= 0)
    vt.require(p.a + p.b == 1.0, tol=1e-6)
    vt.require(p.b <= 0.8)


class Portfolio(BaseModel):
    weights: dict
    expected_return: float
    volatility: float
    sharpe_ratio: float


COV = {
    ("AAPL", "AAPL"): 0.048,
    ("AAPL", "MSFT"): 0.031,
    ("AAPL", "NVDA"): 0.042,
    ("MSFT", "MSFT"): 0.039,
    ("MSFT", "NVDA"): 0.036,
    ("NVDA", "NVDA"): 0.095,
}
RET = {"AAPL": 0.14, "MSFT": 0.13, "NVDA": 0.24}


@vt.verified_schema()
def portfolio(p: Portfolio):
    vt.require(vt.keys_subset(p.weights, {"AAPL", "MSFT", "NVDA"}))
    vt.require(vt.sum(p.weights.values()) == 1.0, tol=1e-6)
    vt.require(vt.forall(p.weights.values(), lambda w: (0.05 <= w) & (w <= 0.50)))
    vt.derive(p.expected_return == vt.dot(p.weights, RET), tol=1e-5)
    vt.derive(p.volatility == vt.sqrt(vt.quad_form(p.weights, COV)), tol=1e-5)
    vt.derive(p.sharpe_ratio == (p.expected_return - 0.04) / p.volatility, tol=1e-5)


def _weights():
    return {"AAPL": 0.40, "MSFT": 0.35, "NVDA": 0.25}


def test_verify_pass():
    report = linear.verify({"a": 0.3, "b": 0.7})
    assert report.status == "PASS"
    assert report.ok
    assert report.missing == []
    assert isinstance(report.model, Linear)


def test_verify_fail_has_core_and_residual():
    report = linear.verify({"a": 0.3, "b": 0.9})
    assert report.status == "FAIL"
    assert report.unsat_core, "a violated constraint set must be attributed"
    violated = {o.text for o in report.violations}
    assert any("a + b" in text for text in violated)
    eq = [o for o in report.violations if "a + b" in o.text][0]
    assert abs(eq.residual - 0.2) < 1e-9
    assert report.relaxation


def test_verify_incomplete_is_never_pass():
    report = linear.verify({"a": 0.5})
    assert report.status == "INCOMPLETE"
    assert "b" in report.missing


def test_derived_fields_are_filled_exactly():
    report = portfolio.verify(
        {"weights": _weights(), "expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}
    )
    # Derived fields were supplied (wrongly); fill_derived must not silently overwrite
    # them, so the equality constraints are reported as violations instead.
    assert report.status == "FAIL"
    assert any("expected_return" in o.text for o in report.violations)

    filled = portfolio.derive(weights=_weights())
    expected = vt.dot(_weights(), RET)
    volatility = vt.sqrt(vt.quad_form(_weights(), COV))
    assert abs(filled["expected_return"] - round(expected, 5)) < 1e-9
    assert abs(filled["volatility"] - round(volatility, 5)) < 1e-9
    assert abs(filled["sharpe_ratio"] - round((expected - 0.04) / volatility, 5)) < 1e-9


def test_missing_derived_fields_are_completed_by_verify():
    report = portfolio.verify({"weights": _weights()})
    assert report.status == "PASS", report.summary_line()
    assert set(report.derived) == {"expected_return", "volatility", "sharpe_ratio"}


def test_container_element_bounds_are_enforced():
    bad = {"AAPL": 0.40, "MSFT": 0.55, "NVDA": 0.05}
    report = portfolio.verify({"weights": bad})
    assert report.status == "FAIL"
    # The forall element bound must be the violated clause.
    assert any("forall" in o.text for o in report.violations)


def test_quad_form_is_symmetric():
    w = _weights()
    direct = 0.0
    labels = list(w.keys())
    for _i, a in enumerate(labels):
        for _j, b in enumerate(labels):
            key = (a, b) if (a, b) in COV else (b, a)
            direct += w[a] * COV[key] * w[b]
    assert abs(vt.quad_form(w, COV) - direct) < 1e-12


def test_spec_serialization_roundtrip_is_stable():
    payload = spec_to_dict(portfolio.spec)
    clone = spec_from_dict(payload)
    assert clone.field_order == portfolio.spec.field_order
    assert spec_digest(clone) == spec_digest(portfolio.spec)
    assert [r.target_json_key for r in clone.zero_dof_rules] == [
        r.target_json_key for r in portfolio.spec.zero_dof_rules
    ]
    report = verify_values(clone, {"weights": _weights()})
    assert report.status == "PASS", report.summary_line()


def test_describe_has_actions():
    described = portfolio.describe()
    assert described["derived_fields"] == ["expected_return", "volatility", "sharpe_ratio"]
    assert described["containers"][0]["universe"] == ["AAPL", "MSFT", "NVDA"]
    assert described["digest"]


def test_json_instance():
    payload = json.dumps({"a": 0.25, "b": 0.75})
    assert linear.verify(payload).status == "PASS"
