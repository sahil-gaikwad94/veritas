"""End-to-end solver-guided decoding with deterministic reference models."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest
from pydantic import BaseModel

import vexx as vt
from vexx.decode import DecodeConfig, VexxSession
from vexx.json_pda import JSONPushdownAutomaton, is_complete_json
from vexx.testing import MockCharModel, RandomCharModel


class Linear(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def linear(p: Linear):
    vt.require(p.a >= 0)
    vt.require(p.b >= 0)
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


class Product(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def nonlinear(p: Product):
    vt.require(p.a >= 0)
    vt.require(p.b >= 0)
    vt.require(p.a <= 1)
    vt.require(p.b <= 1)
    # Deliberately not affine: the contractor cannot propagate this, so the engine must
    # rely on clause verification + CDCL domain cuts.
    vt.require(p.a * p.b <= 0.1, tol=1e-6)


def test_linear_generation_is_valid_and_constrained():
    session = VexxSession(linear, MockCharModel(), DecodeConfig())
    result = session.run(prompt="pick two weights that sum to one")
    assert result.ok, result.report.summary_line()
    assert is_complete_json(result.text)
    parsed = json.loads(result.text)
    assert abs(parsed["a"] + parsed["b"] - 1.0) < 1e-6
    assert 0 <= parsed["b"] <= 0.8 + 1e-9
    assert result.telemetry.solver_checks >= 1
    assert result.telemetry.masked_candidates > 0
    assert result.report.model is not None


def test_container_residual_accumulator_and_zero_dof_splice():
    session = VexxSession(portfolio, MockCharModel())
    result = session.run()
    assert result.ok, result.report.summary_line()
    parsed = json.loads(result.text)
    weights = parsed["weights"]
    assert set(weights).issubset({"AAPL", "MSFT", "NVDA"})
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(0.05 <= w <= 0.50 for w in weights.values())

    # Derived fields must be exact (within precision), not sampled.
    expected = sum(weights[k] * RET[k] for k in weights)
    variance = 0.0
    labels = list(weights.keys())
    for _i, a in enumerate(labels):
        for _j, b in enumerate(labels):
            key = (a, b) if (a, b) in COV else (b, a)
            variance += weights[a] * COV[key] * weights[b]
    assert abs(parsed["expected_return"] - expected) < 1e-4
    assert abs(parsed["volatility"] - variance ** 0.5) < 1e-4
    assert result.telemetry.jump_forward_fields == 3
    assert result.telemetry.jump_forward_chars > 0


def test_nonlinear_conflict_triggers_domain_cuts():
    config = DecodeConfig(max_backtracks=8)
    session = VexxSession(nonlinear, MockCharModel(), config)
    result = session.run()
    assert result.ok, result.report.summary_line()
    parsed = json.loads(result.text)
    assert parsed["a"] * parsed["b"] <= 0.1 + 1e-6
    assert result.telemetry.backtracks >= 1
    assert result.telemetry.domain_cuts
    assert any("bounds" in cut for cut in result.telemetry.domain_cuts)


def test_generate_returns_verified_result_and_proof():
    result = vt.generate(
        model=MockCharModel(),
        prompt="sum to one",
        spec=linear,
        return_proof=True,
    )
    assert result.ok
    assert isinstance(result.value, Linear)
    assert result.proof is not None
    verification = vt.verify_proof(result.proof)
    assert verification.ok, verification.as_dict()
    assert "(check-sat)" in result.proof.smt2
    assert result.telemetry["final_status"] == "PASS"


def test_generate_requires_a_model_and_spec():
    with pytest.raises(ValueError):
        vt.generate(model=None, spec=linear)
    with pytest.raises(ValueError):
        vt.generate(model=MockCharModel(), spec=None)


def test_random_model_still_produces_valid_json_or_incomplete():
    # A random model is allowed to fail constraints, but the engine must always emit a
    # syntactically valid document and report the outcome honestly.
    session = VexxSession(linear, RandomCharModel(seed=7), DecodeConfig(max_backtracks=2))
    result = session.run()
    assert is_complete_json(result.text)
    assert result.report.status in ("PASS", "FAIL", "INCOMPLETE")
    if result.ok:
        parsed = json.loads(result.text)
        assert abs(parsed["a"] + parsed["b"] - 1) < 1e-6


def test_grammar_always_valid_json():
    machine = JSONPushdownAutomaton()
    assert machine.feed('{"a": 1}')
    assert machine.accepts()


def test_engine_respects_explicit_bounds():
    class Bounded(BaseModel):
        x: float

    @vt.verified_schema()
    def bounded(p: Bounded):
        vt.require(p.x >= 0.2)
        vt.require(p.x <= 0.3)

    session = VexxSession(bounded, MockCharModel())
    result = session.run()
    assert result.ok, result.report.summary_line()
    value = json.loads(result.text)["x"]
    assert 0.2 - 1e-9 <= value <= 0.3 + 1e-9
    # The high digits the model prefers must have been masked away.
    assert result.telemetry.contractor_prunes > 0
