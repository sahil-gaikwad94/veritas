"""Deep cartridge tests: real algorithms, not surface shims."""

import math

import pytest

from veritas.cartridges.chem import (
    analyze_smiles,
    formal_charge_neutral,
    molecular_weight,
    smiles_valence_ok,
)
from veritas.cartridges.code_synth import verify_python_source
from veritas.cartridges.or_sched import SchedulingError, SchedulingProblem
from veritas.cartridges.phys import (
    ACCELERATION,
    ENERGY,
    FORCE,
    MASS,
    TIME,
    VELOCITY,
    Quantity,
    check_equation,
)
from veritas.cartridges.quant import (
    black_scholes_call,
    cholesky,
    gaap_balance,
    implied_volatility,
    is_positive_semidefinite,
    portfolio_stats,
    portfolio_weights_valid,
    quad_form,
)
from veritas.cartridges.sql import ForeignKey, SqlCatalog, SqlVerifier
from veritas.cartridges.stat import (
    bayes_consistency,
    frechet_ok,
    is_d_separated,
    probability_simplex,
)
from veritas.pda import PushdownAutomaton
from veritas.shadow_ast import ShadowParser


def test_portfolio_statistics_and_cholesky():
    weights = {"A": 0.5, "B": 0.5}
    returns = {"A": 0.10, "B": 0.20}
    cov = {("A", "A"): 0.04, ("B", "B"): 0.09, ("A", "B"): 0.02}
    stats = portfolio_stats(weights, returns, cov)
    assert abs(stats.expected_return - 0.15) < 1e-12
    assert abs(stats.variance - 0.0425) < 1e-12
    assert abs(stats.volatility - math.sqrt(0.0425)) < 1e-12
    assert abs(stats.sharpe_ratio - (0.15 - 0.04) / math.sqrt(0.0425)) < 1e-12
    assert abs(quad_form(weights, cov) - 0.0425) < 1e-12
    lower = cholesky([[4.0, 2.0], [2.0, 3.0]])
    assert lower is not None
    assert is_positive_semidefinite([[4.0, 2.0], [2.0, 3.0]])
    assert not is_positive_semidefinite([[1.0, 2.0], [2.0, 1.0]])
    assert portfolio_weights_valid(weights, allowed={"A", "B"})
    assert gaap_balance(100.0, 60.0, 40.0)


def test_implied_volatility_roundtrip():
    price = black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)
    implied = implied_volatility(price, 100.0, 100.0, 0.05, 1.0)
    assert implied is not None
    assert abs(implied - 0.2) < 1e-6
    assert implied_volatility(0.0, 100.0, 100.0, 0.05, 1.0) is None


def test_sql_catalog_alias_fk_and_tenant():
    catalog = SqlCatalog(
        tables={
            "users": ["id", "tenant_id", "name"],
            "orders": ["id", "tenant_id", "user_id", "total"],
        },
        foreign_keys=[ForeignKey("orders", "user_id", "users", "id")],
        require_tenant=True,
    )
    verifier = SqlVerifier(catalog)
    good = verifier.verify(
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id "
        "WHERE u.tenant_id = :tenant"
    )
    assert good.ok, good.errors
    assert good.tables == ["orders", "users"]
    assert "users.name" in good.columns and "orders.total" in good.columns
    assert good.has_tenant_filter

    unknown = verifier.verify("SELECT u.email FROM users u WHERE u.tenant_id = 1")
    assert not unknown.ok
    assert any("email" in error for error in unknown.errors)

    bad_fk = verifier.verify(
        "SELECT u.id FROM users u JOIN orders o ON u.id = o.id WHERE u.tenant_id = 1"
    )
    assert not bad_fk.ok
    assert any("foreign key" in error for error in bad_fk.errors)

    no_tenant = verifier.verify("SELECT u.name FROM users u")
    assert not no_tenant.ok
    assert any("tenant" in error for error in no_tenant.errors)

    star = verifier.verify("SELECT u.* FROM users u WHERE u.tenant_id = 1")
    assert star.ok, star.errors

    destructive = verifier.verify("DELETE FROM users WHERE tenant_id = 1")
    assert not destructive.ok
    assert any("read-only" in error for error in destructive.errors)

    assert verifier.verify("SELECT * FROM missing").errors


def test_scheduling_solver_cycle_and_capacity():
    problem = SchedulingProblem()
    problem.add_task("A", 2)
    problem.add_task("B", 3, predecessors=["A"])
    problem.add_task("C", 1, predecessors=["B"], deadline=10)
    starts = problem.earliest_schedule()
    assert starts == {"A": 0.0, "B": 2.0, "C": 5.0}
    assert problem.validate(starts) == []
    assert problem.makespan() == 6.0
    assert problem.critical_path() == (6.0, ["A", "B", "C"])

    cyclic = SchedulingProblem()
    cyclic.add_task("X", 1, predecessors=["Y"])
    cyclic.add_task("Y", 1, predecessors=["X"])
    with pytest.raises(SchedulingError):
        cyclic.topological_order()

    capacity = SchedulingProblem()
    capacity.add_task("P", 2, resource="cpu")
    capacity.add_task("Q", 2, resource="cpu")
    capacity.set_capacity("cpu", 1)
    violations = capacity.validate(capacity.earliest_schedule())
    assert any(v.kind == "capacity" for v in violations)


def test_probability_and_d_separation():
    assert probability_simplex([0.2, 0.3, 0.5])
    assert not probability_simplex([0.2, 0.3, 0.6])
    assert frechet_ok(0.5, 0.4, 0.2)
    assert not frechet_ok(0.5, 0.4, 0.9)
    assert bayes_consistency(0.5, 0.4, 0.2, 1.0)
    chain = {"m": ["x"], "y": ["m"]}
    assert not is_d_separated(chain, "x", "y")
    assert is_d_separated(chain, "x", "y", "m")
    collider = {"c": ["x", "y"]}
    assert is_d_separated(collider, "x", "y")
    assert not is_d_separated(collider, "x", "y", "c")


def test_dimensional_analysis_equations():
    symbols = {"m": MASS, "a": ACCELERATION, "F": FORCE, "E": ENERGY,
               "c": VELOCITY, "v": VELOCITY, "t": TIME}
    ok, message = check_equation("F", "m * a", symbols)
    assert ok, message
    ok, message = check_equation("E", "m * c ** 2", symbols)
    assert ok, message
    ok, message = check_equation("F", "m * v", symbols)
    assert not ok and "mismatch" in message
    ok, message = check_equation("v", "a * t", symbols)
    assert ok, message
    ok, message = check_equation("sin(v)", "0", symbols)
    assert not ok
    from veritas.cartridges.phys import Dim

    q = Quantity(2.0, Dim(FORCE))
    assert q.dim.dimensionless is False


def test_smiles_valence_and_formula():
    ethanol = analyze_smiles("CCO")
    assert ethanol.ok, ethanol.errors
    assert ethanol.formula == {"C": 2, "H": 6, "O": 1}
    assert abs((molecular_weight("CCO") or 0) - 46.07) < 0.1
    benzene = analyze_smiles("c1ccccc1")
    assert benzene.ok, benzene.errors
    assert benzene.formula == {"C": 6, "H": 6}
    assert analyze_smiles("C=O").ok
    assert analyze_smiles("C#N").ok
    assert not smiles_valence_ok("C(C)(C)(C)(C)C")
    assert not analyze_smiles("C1CC").ok
    assert not analyze_smiles("C(").ok
    salt = analyze_smiles("[Na+].[Cl-]")
    assert salt.ok, salt.errors
    assert salt.formal_charge == 0
    assert formal_charge_neutral([1, -1])
    assert not formal_charge_neutral([1, 1])


def test_code_guardrails():
    good = verify_python_source(
        "def f(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i\n"
        "    return total\n"
    )
    assert good.ok, good.errors
    undefined = verify_python_source("def f():\n    return missing_name\n")
    assert not undefined.ok and undefined.errors[0].kind == "undefined_identifier"
    divide = verify_python_source("def f():\n    return 1 / 0\n")
    assert not divide.ok and divide.errors[0].kind == "division_by_zero"
    bounds = verify_python_source("def f():\n    return [1, 2][3]\n")
    assert not bounds.ok and bounds.errors[0].kind == "index_out_of_bounds"
    comp = verify_python_source("def f(xs):\n    return [x * 2 for x in xs]\n")
    assert comp.ok, comp.errors


def test_shadow_parser_and_token_mask():
    parser = ShadowParser()
    assert [clause.key for clause in parser.feed_bytes('{"a": 1, "b": 2')] == ["a"]
    clauses = parser.feed_bytes("}")
    assert [clause.key for clause in clauses] == ["b"]
    assert parser.parsed_value("a") == 1

    nested = ShadowParser()
    clauses = nested.feed_bytes('{"a": {"x": 1}, "b": [1, 2]}')
    assert [clause.key for clause in clauses] == ["a", "b"]

    pda = PushdownAutomaton(vocab_size=8)
    assert pda.get_valid_token_bitmask() == (1 << 8) - 1
    assert pda.feed('{"a": 1}')
    assert pda.accepts()
    assert not pda.feed("{}")
