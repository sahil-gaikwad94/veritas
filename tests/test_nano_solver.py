"""Phase 3 tests: nano solver."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.nano_solver import SolverStatus, VeritasNanoSolver


def test_lra_unsat_core():
    s = VeritasNanoSolver()
    s.assert_and_track(("le", {"x": 1.0}, 3.0), "c1")  # x <= 3
    s.assert_and_track(("ge", {"x": 1.0}, 5.0), "c2")  # x >= 5
    assert s.check() == SolverStatus.UNSAT
    core = s.get_unsat_core()
    assert "c1" in core and "c2" in core


def test_push_pop():
    s = VeritasNanoSolver()
    s.push()
    s.assert_and_track(("le", {"x": 1.0}, 3.0), "c1")
    s.pop()
    s.assert_and_track(("ge", {"x": 1.0}, 5.0), "c2")
    # after pop, first constraint gone; single constraint feasible
    assert s.check() == SolverStatus.SAT


def test_diff_logic_negative_cycle():
    s = VeritasNanoSolver()
    # x - y <= 5, y - z <= 5, z - x <= -20  (cycle)
    s.assert_and_track(("diff", "x", "y", 5), "e1")
    s.assert_and_track(("diff", "y", "z", 5), "e2")
    s.assert_and_track(("diff", "z", "x", -20), "e3")
    assert s.check() == SolverStatus.UNSAT
    core = s.get_unsat_core()
    assert len(core) >= 1


def test_sat_simple():
    s = VeritasNanoSolver()
    s.assert_and_track(("le", {"x": 1.0, "y": 1.0}, 1.0), "c1")
    s.assert_and_track(("ge", {"x": 1.0}, 0.0), "c2")
    s.assert_and_track(("ge", {"y": 1.0}, 0.0), "c3")
    assert s.check() == SolverStatus.SAT
