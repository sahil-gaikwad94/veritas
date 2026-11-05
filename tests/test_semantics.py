"""Semantic regressions: strictness, nulls, schema branches, coverage, cartridges."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

import vexx as vt
from vexx.bridges.llamacpp_bridge import patch_llamacpp
from vexx.cartridges.chem import molecular_weight
from vexx.cartridges.code_synth import verify_python_source
from vexx.cartridges.or_sched import SchedulingProblem
from vexx.cartridges.quant import is_positive_semidefinite
from vexx.interval import Hc4AffineContractor, Interval
from vexx.json_pda import JSONSyntaxError
from vexx.proof.certificate import build_certificate, verify_certificate
from vexx.schema_pda import SchemaJSONAutomaton
from vexx.serde import spec_from_dict, spec_to_dict


class Strict(BaseModel):
    x: float = Field(ge=0.0, le=2.0)


@vt.verified_schema()
def strict(p: Strict):
    vt.require((p.x < 1.0) & (p.x > 0.0))


class Nullable(BaseModel):
    x: float | None


@vt.verified_schema()
def nullable(p: Nullable):
    vt.require(p.x == None)  # noqa: E711 - DSL null comparison


class Root(BaseModel):
    x: float = Field(ge=0.0)


@vt.verified_schema()
def root(p: Root):
    vt.require(vt.sqrt(p.x) >= 0.0)


def test_strict_comparisons_are_exact_by_default():
    assert strict.verify({"x": 0.5}).ok
    assert not strict.verify({"x": 1.0}).ok
    assert not strict.verify({"x": 0.0}).ok


def test_present_null_is_distinct_from_missing():
    assert nullable.verify({"x": None}).ok
    assert not nullable.verify({}).ok


def test_serde_rejects_malformed_payloads():
    payload = spec_to_dict(strict.spec)
    bad_version = dict(payload, version=2)
    with pytest.raises(ValueError):
        spec_from_dict(bad_version)
    bad_tol = spec_to_dict(strict.spec)
    bad_tol["requires"][0]["tol"] = -1.0
    with pytest.raises(ValueError):
        spec_from_dict(bad_tol)
    bad_vars = spec_to_dict(strict.spec)
    bad_vars["variables"]["not-an-id"] = {"name": "x", "type": None}
    with pytest.raises(ValueError):
        spec_from_dict(bad_vars)


def test_schema_one_of_branches_do_not_merge():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"a": {"type": "integer"}},
             "required": ["a"], "additionalProperties": False},
            {"type": "object", "properties": {"b": {"type": "string"}},
             "required": ["b"], "additionalProperties": False},
        ]
    }
    good = SchemaJSONAutomaton(schema)
    assert good.feed('{"a": 1}')
    assert good.finish()
    bad = SchemaJSONAutomaton(schema)
    assert not (bad.feed('{"a": 1, "b": "x"}') and bad.finish())


def test_schema_recursive_ref_is_rejected():
    with pytest.raises(JSONSyntaxError):
        SchemaJSONAutomaton({"$ref": "#"})


def test_certificate_smt_coverage_is_explicit_and_bound():
    cert = build_certificate(root, {"x": 0.25})
    assert cert.smt2_complete is False
    assert cert.smt2_omitted
    assert verify_certificate(cert).ok
    cert.smt2_complete = True
    cert.seal()
    verification = verify_certificate(cert)
    assert not verification.ok
    assert any(check.name == "smt2_coverage" and not check.ok for check in verification.checks)


def test_interval_negative_coefficient_inequalities_are_sound():
    contractor = Hc4AffineContractor({0: Interval(0, 10), 1: Interval(0, 10)})
    contractor.add_constraint(("linear", {0: -1.0, 1: 1.0}, 0.0, "le"))
    assert contractor.propagate()
    assert contractor.get_bounds(0).lo == 0.0  # x >= y, not x >= 10

    contractor = Hc4AffineContractor({0: Interval(0, 10), 1: Interval(0, 10)})
    contractor.add_constraint(("linear", {0: -1.0, 1: 1.0}, 0.0, "ge"))
    assert contractor.propagate()
    assert contractor.get_bounds(0).hi == 10.0  # x <= y, not x <= 0


def test_psd_accepts_singular_but_not_indefinite():
    assert is_positive_semidefinite([[1.0, 1.0], [1.0, 1.0]])
    assert not is_positive_semidefinite([[1.0, 2.0], [2.0, 1.0]])
    assert not is_positive_semidefinite([[-1.0]])


def test_critical_path_includes_release_times():
    problem = SchedulingProblem()
    problem.add_task("A", 1, release=5)
    problem.add_task("B", 2, predecessors=["A"])
    assert problem.critical_path() == (8.0, ["A", "B"])


def test_unknown_element_mass_is_not_zero():
    assert molecular_weight("[Xx]") is None


def test_llamacpp_grammar_only_host_fails_closed_with_spec():
    model = SimpleNamespace(grammar="")
    with pytest.raises(TypeError):
        patch_llamacpp(model, strict, None)


def test_code_synth_walrus_match_and_global():
    assert verify_python_source("y = (x := 1) + x").ok
    source = """
def f(value):
    match value:
        case {"k": v}:
            return v
        case _:
            return 0
"""
    assert verify_python_source(source).ok
    report = verify_python_source("def f():\n    global g\n    return g\n")
    assert report.ok
    assert any(w.kind == "global_binding" for w in report.warnings)

def test_verify_json_reports_grammar_errors_as_format_errors():
    from vexx.instance import InstanceFormatError, verify_json
    with pytest.raises(InstanceFormatError):
        verify_json(strict, '{"x": 01}')


def test_synth_sanitizes_identifiers_and_is_deterministic():
    import ast

    from vexx.synth import synth_schema

    samples = [{"bad-name": 1, "class": 2, "UPPER": 3}]
    source = synth_schema(samples, name="Demo")
    ast.parse(source)
    assert "bad_name" in source
    assert "field_class" in source
    assert synth_schema(samples, name="Demo") == source


def test_quad_form_matrix_path_does_not_require_numpy(monkeypatch):
    import sys

    # Match the core CI extra set: importing NumPy must fail on this path.
    monkeypatch.setitem(sys.modules, "numpy", None)
    matrix = [
        [0.048, 0.031, 0.042],
        [0.031, 0.039, 0.036],
        [0.042, 0.036, 0.095],
    ]
    assert vt.quad_form([0.4, 0.35, 0.25], matrix) == pytest.approx(0.041775)
    assert vt.quad_form([1.0, 2.0], [[1.0, 0.0]]) is None
