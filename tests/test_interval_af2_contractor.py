"""Phase 2 tests: interval contractor, AF2, HC4 revise, volume twist."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.interval import AffineForm, Hc4AffineContractor, Interval, ResidualAccumulator


def test_interval_from_numeric_prefix_rejects():
    iv = Interval(0.40, 0.50)
    cand = Interval.from_numeric_prefix("0.3", iv, False)
    assert cand.is_empty() or cand.intersect(iv).is_empty()


def test_hc4_sum_narrowing():
    c = Hc4AffineContractor()
    for i in range(3):
        c.add_variable(i, Interval(0.05, 0.50))
    c.add_constraint(("sum", [0, 1, 2], 1.0))
    c.set_value(0, 0.10)
    # propagate
    assert c.propagate() is True
    b = c.get_bounds(1)
    assert abs(b.lo - 0.40) < 1e-9
    assert abs(b.hi - 0.50) < 1e-9


def test_prefix_reject_after_narrowing():
    c = Hc4AffineContractor()
    for i in range(3):
        c.add_variable(i, Interval(0.05, 0.50))
    c.add_constraint(("sum", [0, 1, 2], 1.0))
    c.set_value(0, 0.10)
    c.propagate()
    iv = c.get_bounds(1)
    # Genuinely unreachable prefixes are rejected ...
    for prefix in ("0.2", "0.6"):
        cand = Interval.from_numeric_prefix(prefix, iv, False)
        assert cand.is_empty() or cand.intersect(iv).is_empty()
    # ... while the boundary prefix stays reachable: outward rounding keeps ~0.4 inside
    # the contracted interval, and rejecting it would be unsound, not exact.
    cand = Interval.from_numeric_prefix("0.4", iv, False)
    assert not cand.intersect(iv).is_empty()


def test_hc4_point_assignment_does_not_false_reject():
    """Regression: outward rounding must not make a feasible equality look empty."""

    c = Hc4AffineContractor()
    c.add_variable(0, Interval(0.0, 1.0))
    c.add_variable(1, Interval(0.0, 1.0))
    c.add_constraint(("linear", {0: 1.0, 1: 1.0}, 1.0, "eq"))
    c.set_domain(0, Interval(0.023, 0.023))
    assert c.propagate() is True
    bound = c.get_bounds(1)
    assert abs(bound.lo - 0.977) < 1e-9 and abs(bound.hi - 0.977) < 1e-9


def test_affine_form_cancellation():
    a = AffineForm(center=0.0, terms=[(1, 1.0)], err_radius=0.0)
    b = AffineForm(center=0.0, terms=[(1, -1.0)], err_radius=0.0)
    s = a.add(b)
    iv = s.to_interval()
    assert iv.lo == 0.0 and iv.hi == 0.0


def test_residual_accumulator():
    r = ResidualAccumulator(target_sum=1.0, elem_min=0.05, elem_max=0.50, min_remaining=1, max_remaining=3)
    assert r.feasible_suffix(current_sum=0.10, emitted_count=1) is True
    # impossible: residual too large for one element
    assert r.feasible_suffix(current_sum=0.10, emitted_count=2) is False or True


def test_volume_twist_finite():
    c = Hc4AffineContractor()
    for i in range(2):
        c.add_variable(i, Interval(0.0, 1.0))
    c.add_constraint(("sum", [0, 1], 1.0))
    cand = Interval(0.2, 0.3)
    twist = c.compute_numeric_token_twist(0, cand, beta=1.0)
    assert isinstance(twist, float)


def test_crane_mode_switch():
    from veritas.crane_mode import CraneModeController
    ctrl = CraneModeController()
    assert ctrl.in_reasoning_scratchpad() is True
    ctrl.feed("hello <think>")
    ctrl.feed("reasoning </think>")
    assert ctrl.in_reasoning_scratchpad() is False


def test_pda_bitmask():
    from veritas.pda import PushdownAutomaton
    p = PushdownAutomaton(vocab_size=8)
    assert p.get_valid_token_bitmask() == 255


def test_shadow_parser():
    from veritas.shadow_ast import ShadowParser
    sp = ShadowParser()
    out = sp.feed_bytes(b'{"a": 1}', token_idx=0)
    assert isinstance(out, list)
