"""Tests for ``veritas explain-unsat`` (minimal relaxation wizard)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pydantic import BaseModel, Field

import veritas as vt
from veritas.cli.main import main
from veritas.elastic import explain_unsat


def test_bound_conflict_reported_exactly(tmp_path, capsys):
    path = tmp_path / "bound_unsat.py"
    path.write_text(
        "import veritas as vt\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class P(BaseModel):\n"
        "    weight: float = Field(ge=0.0, le=0.25)\n"
        "\n"
        "@vt.verified_schema()\n"
        "def bound_unsat(p: P):\n"
        "    vt.require(p.weight >= 0.4)\n",
        encoding="utf-8",
    )
    code = main(["explain-unsat", str(path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "bound" in out.lower() or "weight" in out


def test_clause_conflict_gets_minimal_slack():
    class P(BaseModel):
        a: float = Field(ge=0.0, le=1.0)
        b: float = Field(ge=0.0, le=1.0)

    @vt.verified_schema(preflight_check=False)
    def clause_unsat(p: P):
        vt.require(p.a + p.b == 1.0, tol=1e-6)
        vt.require(p.a >= 0.7)
        vt.require(p.b >= 0.7)

    report = explain_unsat(clause_unsat.spec)
    assert report["status"] == "UNSAT"
    eq_suggestions = [s for s in report["suggestions"] if s["op"] == "eq"]
    assert eq_suggestions, report["suggestions"]
    assert eq_suggestions[0]["delta"] < 1.0  # minimal widening exists
    assert eq_suggestions[0]["delta"] > 0.0


def test_clause_conflict_cli_exit_codes(tmp_path):
    path = tmp_path / "clause_unsat.py"
    path.write_text(
        "import veritas as vt\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class P(BaseModel):\n"
        "    a: float = Field(ge=0.0, le=1.0)\n"
        "    b: float = Field(ge=0.0, le=1.0)\n"
        "\n"
        "@vt.verified_schema()\n"
        "def clause_unsat(p: P):\n"
        "    vt.require(p.a + p.b == 1.0, tol=1e-6)\n"
        "    vt.require(p.a >= 0.7)\n"
        "    vt.require(p.b >= 0.7)\n",
        encoding="utf-8",
    )
    assert main(["explain-unsat", str(path)]) == 1


def test_sat_spec_exits_zero(tmp_path, capsys):
    path = tmp_path / "sat.py"
    path.write_text(
        "import veritas as vt\n"
        "from pydantic import BaseModel, Field\n"
        "\n"
        "class P(BaseModel):\n"
        "    a: float = Field(ge=0.0, le=1.0)\n"
        "    b: float = Field(ge=0.0, le=1.0)\n"
        "\n"
        "@vt.verified_schema()\n"
        "def ok(p: P):\n"
        "    vt.require(p.a + p.b == 1.0, tol=1e-6)\n"
        "    vt.require(p.a >= 0.0)\n"
        "    vt.require(p.b >= 0.0)\n",
        encoding="utf-8",
    )
    assert main(["explain-unsat", str(path)]) == 0


def test_all_repairs_need_structural_fix_reported():
    class P(BaseModel):
        a: float = Field(ge=0.0, le=1.0)
        b: float = Field(ge=0.0, le=1.0)

    @vt.verified_schema(preflight_check=False)
    def double_conflict(p: P):
        vt.require(p.a + p.b == 1.0, tol=1e-6)
        vt.require(p.a >= 0.7)
        vt.require(p.b >= 0.7)
        vt.require(p.a <= 0.1)  # second, independent conflict

    report = explain_unsat(double_conflict.spec)
    assert report["status"] == "UNSAT"
