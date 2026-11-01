"""Phase 1 tests: topological AST-to-VIR compilation and pre-flight gate."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pydantic import BaseModel, Field

import veritas as vt


class SimpleModel(BaseModel):
    a: float = Field(default=0.0)
    b: float = Field(default=0.0)


def test_basic_verified_schema_compiles():
    @vt.verified_schema()
    def spec(p: SimpleModel):
        vt.require(p.a >= 0)
        vt.require(p.b <= 10)

    assert spec.spec is not None
    assert set(spec.field_order) == {"a", "b"}
    assert spec.preflight.is_sat is True


class PortfolioAllocation(BaseModel):
    weights: dict = Field(default_factory=dict)
    expected_return: float = Field(default=0.0)
    volatility: float = Field(default=0.0)
    sharpe_ratio: float = Field(default=0.0)


COV_MATRIX = {
    ("AAPL", "AAPL"): 0.048, ("AAPL", "MSFT"): 0.031, ("AAPL", "NVDA"): 0.042,
    ("MSFT", "MSFT"): 0.039, ("MSFT", "NVDA"): 0.036, ("NVDA", "NVDA"): 0.095,
}
ASSET_RETURNS = {"AAPL": 0.14, "MSFT": 0.13, "NVDA": 0.24}


def test_portfolio_topological_order():
    @vt.verified_schema()
    def valid_portfolio(p: PortfolioAllocation):
        vt.require(vt.keys_subset(p.weights, {"AAPL", "MSFT", "NVDA"}))
        vt.require(vt.sum(p.weights.values()) == 1.0, tol=1e-6)
        vt.require(vt.forall(p.weights.values(), lambda w: (0.05 <= w) & (w <= 0.50)))
        vt.derive(p.expected_return == vt.dot(p.weights, ASSET_RETURNS), tol=1e-5)
        vt.derive(p.volatility == vt.sqrt(vt.quad_form(p.weights, COV_MATRIX)), tol=1e-5)
        vt.derive(p.sharpe_ratio == (p.expected_return - 0.04) / p.volatility, tol=1e-5)
        vt.require(p.volatility <= 0.21)
        vt.require(p.sharpe_ratio >= 0.65)

    order = valid_portfolio.field_order
    # weights must come before derived fields
    assert order.index("weights") < order.index("expected_return")
    assert order.index("weights") < order.index("volatility")
    assert order.index("weights") < order.index("sharpe_ratio")
    # sharpe_ratio depends on expected_return & volatility
    assert order.index("expected_return") < order.index("sharpe_ratio")
    assert order.index("volatility") < order.index("sharpe_ratio")


def test_circular_derivation_raises():
    class M(BaseModel):
        x: float = 0.0
        y: float = 0.0

    try:
        @vt.verified_schema()
        def bad(p: M):
            vt.derive(p.x == p.y + 1)
            vt.derive(p.y == p.x + 1)
        raised = False
    except Exception as e:
        raised = True
        assert "Cycle" in str(e) or "cycle" in str(e).lower()
    # CircularDerivationError should have been raised during decoration
    assert raised is True


def test_preflight_unsat_simple():
    class M(BaseModel):
        x: float = 0.0

    # infeasible bounds

    class Infeasible(Exception):
        pass

    raised = False
    try:
        @vt.verified_schema()
        def bad(p: M):
            vt.require(p.x >= 5)
            vt.require(p.x <= 3)
    except Exception as e:
        raised = True
        assert "infeasible" in str(e).lower() or "Infeasible" in str(e)
    assert raised is True


def test_preflight_result_sat():
    class M(BaseModel):
        x: float = 0.0

    @vt.verified_schema(preflight_check=True)
    def ok(p: M):
        vt.require(p.x >= 0)
        vt.require(p.x <= 10)

    assert ok.preflight.status == "SAT"


def test_zero_dof_rules_present():
    @vt.verified_schema()
    def vp(p: PortfolioAllocation):
        vt.derive(p.expected_return == vt.dot(p.weights, ASSET_RETURNS), tol=1e-5)
        vt.derive(p.volatility == vt.sqrt(vt.quad_form(p.weights, COV_MATRIX)), tol=1e-5)

    rules = vp.spec.zero_dof_rules
    targets = {r.target_json_key for r in rules}
    assert "expected_return" in targets
    assert "volatility" in targets


def test_quad_form_cholesky():
    # The compiler's Cholesky path is pure Python; NumPy is not a test dependency.
    @vt.verified_schema()
    def vp(p: PortfolioAllocation):
        vt.derive(p.volatility == vt.sqrt(vt.quad_form(p.weights, COV_MATRIX)), tol=1e-5)

    expected = [
        [0.048, 0.031, 0.042],
        [0.031, 0.039, 0.036],
        [0.042, 0.036, 0.095],
    ]
    found = False
    for e in vp.spec.derives:
        if e.kind == "compare" and e.rhs is not None and e.rhs.kind == "sqrt":
            inner = e.rhs.args[0] if e.rhs.args else None
            if inner is not None and inner.kind == "quad_form_cholesky":
                found = True
                factor = inner.cholesky_l
                assert len(factor) == len(expected)
                for i, row in enumerate(factor):
                    assert len(row) == len(expected)
                    assert all(value == 0.0 for value in row[i + 1:])
                    for j in range(i + 1):
                        reconstructed = sum(factor[i][k] * factor[j][k] for k in range(j + 1))
                        assert reconstructed == pytest.approx(expected[i][j], abs=1e-12)
    assert found is True
