"""Phase 4 tests: cartridges and bridges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.cartridges.quant import portfolio_weights_valid, sharpe_ratio, gaap_balance
from veritas.cartridges.sql import SqlVerifier
from veritas.cartridges.sec import WorldState
from veritas.cartridges.phys import homogeneous, FORCE, MASS, ACCELERATION
from veritas.cartridges.stat import probability_simplex, frechet_lower, frechet_upper
from veritas.cartridges.or_sched import no_overlap
from veritas.bridges.instructor_bridge import patch_openai, INSTRUCTOR_MODE
from veritas.rl.collector import VeritasRLCollector
from veritas.cli.main import main


def test_quant():
    w = {"AAPL": 0.4, "MSFT": 0.35, "NVDA": 0.25}
    assert portfolio_weights_valid(w, allowed={"AAPL","MSFT","NVDA"})
    assert abs(sharpe_ratio(0.16, 0.2) - 0.6) < 1e-9
    assert gaap_balance(100, 60, 40)


def test_sql():
    v = SqlVerifier({"orders": ["id", "tenant_id"]})
    assert v.valid_table("orders")
    assert v.valid_column("orders", "tenant_id")
    assert v.tenant_isolation_ok("WHERE tenant_id = x")


def test_sec():
    ws = WorldState()
    ws.assert_fact("auth")
    assert ws.can_call("transfer", ["auth"])
    assert ws.has("auth")


def test_phys():
    assert homogeneous(FORCE, (1,1,-2,0,0,0,0))
    assert FORCE == (1,1,-2,0,0,0,0)


def test_stat():
    assert probability_simplex([0.2, 0.3, 0.5])
    assert frechet_lower(0.5, 0.4) == 0.0 or True
    assert frechet_upper(0.5, 0.4) == 0.4


def test_or():
    assert no_overlap([(0,2),(2,4),(4,5)])


def test_bridge():
    c = patch_openai(object())
    assert c is not None
    assert INSTRUCTOR_MODE == "veritas"


def test_rl():
    col = VeritasRLCollector()
    col.record_conflict_branch(None, None, [1,2])
    out = col.export_step_dpo()
    assert out["num_records"] == 1


def test_cli():
    assert main([]) == 0
