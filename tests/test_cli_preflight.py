"""CLI end-to-end tests and preflight diagnostics."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import veritas as vt
from veritas.cli.main import load_schemas, main
from veritas.compiler.preflight import InfeasibleSpecError, initial_domains, run_preflight

SCHEMA_SOURCE = '''
import veritas as vt
from pydantic import BaseModel

class Pair(BaseModel):
    a: float
    b: float

@vt.verified_schema()
def pair(p: Pair):
    vt.require(p.a >= 0)
    vt.require(p.b >= 0)
    vt.require(p.a + p.b == 1.0, tol=1e-9)
'''


def _write_schema(tmp_path: Path) -> Path:
    path = tmp_path / "schema_mod.py"
    path.write_text(SCHEMA_SOURCE)
    return path


def test_check_spec_and_verify_cli(tmp_path, capsys):
    schema_path = _write_schema(tmp_path)
    assert main(["check-spec", str(schema_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "pair"
    assert payload[0]["preflight"]["status"] == "SAT"

    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps({"a": 0.25, "b": 0.75}))
    assert main(["verify", str(schema_path), str(instance)]) == 0
    assert "PASS" in capsys.readouterr().out

    instance.write_text(json.dumps({"a": 0.2, "b": 0.2}))
    assert main(["verify", str(schema_path), str(instance)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_verify_proof_cli(tmp_path, capsys):
    schema_path = _write_schema(tmp_path)
    schema = load_schemas(str(schema_path))[0]
    certificate = vt.build_certificate(schema, {"a": 0.25, "b": 0.75})
    proof = certificate.save(tmp_path / "proof.json")
    assert main(["verify-proof", str(proof)]) == 0
    assert "VALID" in capsys.readouterr().out


def test_inspect_cli_writes_html(tmp_path, capsys):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "status": "PASS",
                "telemetry": {"steps": 3, "backtracks": 1, "jump_forward_fields": 2},
                "trace": [
                    {"field": "a", "text": "0.25", "entropy": 0.3},
                    {"field": "b", "text": "0.75", "entropy": 0.1, "kind": "splice"},
                ],
            }
        )
    )
    out = tmp_path / "report.html"
    assert main(["inspect", str(trace), "--out", str(out)]) == 0
    html = out.read_text()
    assert "veritas inspect" in html
    assert "jump_forward_fields" in html
    assert "0.25" in html


def test_synth_cli_generates_importable_schema(tmp_path, capsys):
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps(
            [
                {"name": "alpha", "score": 0.8, "active": True},
                {"name": "beta", "score": 0.6, "active": False},
            ]
        )
    )
    generated = tmp_path / "generated_schema.py"
    assert main(["synth", "--from-json", str(examples), "--out", str(generated)]) == 0
    assert generated.exists()
    # The synthesized module must itself compile and pass preflight.
    assert main(["check-spec", str(generated)]) == 0


def test_preflight_feasible_and_domain_bounds():
    class Box(BaseModel):
        x: float = Field(ge=0.0, le=1.0)
        y: float = Field(ge=0.0, le=1.0)

    @vt.verified_schema()
    def box(p: Box):
        vt.require(p.x + p.y == 1.0, tol=1e-9)

    assert box.preflight.status == "SAT"
    domains = initial_domains(box.spec)
    assert len(domains) == 2


def test_preflight_rejects_interval_infeasibility_with_hints():
    class Bad(BaseModel):
        x: float = Field(ge=0.8, le=1.0)
        y: float = Field(ge=0.8, le=1.0)

    with pytest.raises(InfeasibleSpecError) as error:
        @vt.verified_schema()
        def bad(p: Bad):
            vt.require(p.x + p.y == 1.0, tol=1e-9)

    assert error.value.conflicts
    assert error.value.relaxation["hints"]


def test_preflight_detects_container_infeasibility():
    class Bag(BaseModel):
        weights: dict

    @vt.verified_schema(preflight_check=False)
    def bag(p: Bag):
        vt.require(vt.keys_subset(p.weights, {"A", "B"}))
        vt.require(vt.sum(p.weights.values()) == 1.0, tol=1e-9)
        vt.require(vt.forall(p.weights.values(), lambda w: w <= 0.3))

    result = run_preflight(bag.spec)
    assert result.status == "UNSAT"
    assert result.conflicts
    assert result.relaxation["details"]


# --------------------------------------------------------------------------------------
# `veritas doctor`: the one command that answers "which engine am I running?".
# --------------------------------------------------------------------------------------

def test_doctor_reports_the_reference_engine(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "veritas" in out and "engine" in out
    assert "reference python" in out or "veritas_py" in out
    assert "wire" in out


def test_doctor_json_is_machine_readable(capsys):
    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {"version", "python", "engine", "wire_version", "platform"}
    assert report["python"] in ("pure", "accelerated")
    assert report["engine"]["native_available"] is (report["python"] == "accelerated")
    # The two fields must agree: reporting an engine without a wire version (or vice versa)
    # would be a lie, and this is exactly what an operator pastes into a bug report.
    assert (report["wire_version"] is None) is (report["python"] == "pure")


def test_doctor_strict_fails_when_the_accelerator_is_missing(capsys):
    from veritas import native

    expected = 0 if native.available() else 1
    assert main(["doctor", "--strict"]) == expected
    if expected:
        assert "native extension" in capsys.readouterr().err


def test_doctor_explains_the_kill_switch(monkeypatch, capsys):
    monkeypatch.setenv("VERITAS_NO_NATIVE", "1")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "VERITAS_NO_NATIVE" in out
    assert main(["doctor", "--strict"]) == 1


def test_doctor_subcommand_is_registered(capsys):
    assert main([]) == 0
    help_text = capsys.readouterr().out
    assert "doctor" in help_text
