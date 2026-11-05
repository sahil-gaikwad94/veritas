"""Adversarial regressions for the audited P0/P1 trust boundaries."""

import pytest
from pydantic import BaseModel, Field

import vexx as vt
from vexx.bridges.hf_transformers import VexxLogitsProcessor
from vexx.cartridges.sec import PolicyViolation
from vexx.cartridges.sql import SqlCatalog, SqlVerifier
from vexx.compiler.ast_transpiler import TranspileError
from vexx.json_pda import JSONSyntaxError, TokenMasker, parse_json
from vexx.nano_solver import SolverStatus, VexxNanoSolver
from vexx.proof.certificate import build_certificate, verify_certificate
from vexx.schema_pda import SchemaJSONAutomaton
from vexx.testing import MockCharModel


class Box(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


@vt.verified_schema()
def box(p: Box):
    vt.require(p.x + p.y == 1.0, tol=1e-9)


class Items(BaseModel):
    values: list[float] = Field(min_length=2, max_length=3)


@vt.verified_schema()
def items(p: Items):
    vt.require(vt.forall(p.values, lambda v: (0.0 <= v) & (v <= 1.0)))


def test_compiler_rejects_ignored_statement():
    class M(BaseModel):
        x: float

    with pytest.raises(TranspileError):
        @vt.verified_schema()
        def bad(p: M):
            vt.require(p.x >= 0)
            print("this is not a constraint")


def test_compiler_rejects_unknown_call():
    class M(BaseModel):
        x: float

    with pytest.raises(TranspileError):
        @vt.verified_schema()
        def bad(p: M):
            vt.require(unknown_function(  # noqa: F821 - intentionally rejected by the compiler
p.x) >= 0)


def test_solver_mixed_constraints_unsat_and_core():
    solver = VexxNanoSolver()
    solver.assert_and_track(("le", {0: 1, 1: 1}, 1.0), "sum")
    solver.assert_and_track(("ge", {0: 1}, 0.8), "x")
    solver.assert_and_track(("ge", {1: 1}, 0.8), "y")
    assert solver.check() == SolverStatus.UNSAT
    core = solver.get_unsat_core()
    assert set(core) == {"sum", "x", "y"}


def test_solver_strict_inequality_witness():
    solver = VexxNanoSolver()
    solver.assert_and_track(("gt", {0: 1}, 0.0), "lo")
    solver.assert_and_track(("lt", {0: 1}, 1.0), "hi")
    assert solver.check() == SolverStatus.SAT
    value = solver.get_float_model()[0]
    assert 0.0 < value < 1.0


@pytest.mark.parametrize("text", ["01", "1.", "1e", "1e+", "-", "[1,]", '{"a":1,"a":2}', "NaN", "Infinity", "tru", '{"a":1}x'])
def test_json_rejects_malformed(text):
    with pytest.raises(JSONSyntaxError):
        parse_json(text)


def test_schema_automaton_required_and_union():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["a"],
        "additionalProperties": False,
    }
    ok = SchemaJSONAutomaton(schema)
    assert ok.feed('{"a":1,"b":null}') and ok.accepts()
    bad = SchemaJSONAutomaton(schema)
    assert not (bad.feed('{"b":"x"}') and bad.accepts())
    masker = TokenMasker(["{", '"a"', ":", "1", "}", '"b"', "x"], schema=schema, eos_token=7)
    mask = masker.valid_mask('{"a": 1')
    assert mask[4] is True  # may close: required a present
    assert mask[6] is False


def test_generate_fails_closed_on_dead_end():
    class Impossible(BaseModel):
        x: float = Field(ge=2.0, le=1.0)

    @vt.verified_schema(preflight_check=False)
    def impossible(p: Impossible):
        vt.require(p.x >= 0)

    with pytest.raises(vt.VerificationError):
        vt.generate(model=MockCharModel(), spec=impossible)


def test_typed_list_decoding():
    result = vt.generate(model=MockCharModel(), spec=items)
    assert result.ok
    assert isinstance(result.value.values, list)
    assert 2 <= len(result.value.values) <= 3


def test_certificate_smt_binding_rejects_forged_script():
    cert = build_certificate(box, {"x": 0.25, "y": 0.75})
    data = cert.to_dict()
    data["smt2"] = data["smt2"].replace("(check-sat)", "(check-sat) ; forged")
    forged = type(cert).from_dict(data)
    forged.sha256 = forged.compute_digest()
    verification = verify_certificate(forged)
    assert not verification.ok
    assert any(c.name == "smt2_binding" and not c.ok for c in verification.checks)


def test_sql_rejects_stacked_statements_and_or_tenant():
    catalog = SqlCatalog(tables={"orders": ["id", "tenant_id"]}, require_tenant=True)
    verifier = SqlVerifier(catalog)
    assert not verifier.verify("SELECT id FROM orders WHERE tenant_id = 1; DELETE FROM orders").ok
    assert not verifier.verify("SELECT id FROM orders WHERE tenant_id = 1 OR 1=1").ok
    assert verifier.verify("SELECT id FROM orders WHERE tenant_id = :tenant").ok


def test_world_state_rolls_back_failed_policy():
    state = vt.WorldState()
    state.declare_tool("transfer", cost=10, budget_key="cash")
    state.set_budget("cash", 15)
    state.add_policy(vt.at_most("transfer", 1), name="once")
    state.record_call("transfer")
    with pytest.raises(PolicyViolation):
        state.record_call("transfer")
    assert state.trace.count("transfer") == 1
    assert state.spent.get("cash", 0) == 10


def test_hf_batch_rows_are_isolated():
    class Tok:
        eos_token_id = 2
        def get_vocab(self):
            return {"{": 0, "}": 1, "": 2}
        def decode(self, ids, **kwargs):
            return "".join({0: "{", 1: "}", 2: ""}.get(int(i), "") for i in ids)

    processor = VexxLogitsProcessor(Tok(), spec=None)
    scores = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    processor([[0], [0, 1]], scores)
    # Row 0 is at '{' and may close; row 1 is complete and may emit EOS.
    assert scores[0][1] > float("-inf")
    assert scores[0][2] == float("-inf")
    assert scores[1][2] > float("-inf")