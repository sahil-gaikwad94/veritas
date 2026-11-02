"""Numeric search: soundness of stop decisions and robustness of the DFS.

The decoder decides *when* a numeric literal is finished with ``_number_stop_ok`` and
searches digits with a bounded depth-first walk plus a deterministic fallback.  Those are
exactly the two places where a constraint-guided decoder can go wrong: accepting a value
that violates a clause, or wandering forever when the model's preferred digits are all
infeasible.
"""

import itertools
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import veritas as vt
from veritas.decode import DecodeConfig, VeritasSession
from veritas.interval import Interval


class Box(BaseModel):
    x: float


@vt.verified_schema()
def half_box(p: Box):
    vt.require(0.0 <= p.x)
    vt.require(p.x <= 1.0)
    vt.require(p.x * 2 <= 1.0)


class Ratio(BaseModel):
    a: float
    b: float


@vt.verified_schema()
def ratio_spec(p: Ratio):
    vt.require(p.a >= 0.0)
    vt.require(p.b >= 0.0)
    vt.require(p.a <= 1.0)
    vt.require(p.b <= 1.0)
    vt.require(p.a + p.b == 1.0, tol=1e-6)


CHARSET = "0123456789.-"


def test_every_accepted_stop_point_satisfies_the_constraints():
    """Exhaustive: no prefix the decoder may stop on can violate a clause.

    The interval handed to ``_number_stop_ok`` is the *contracted* one, which is how the
    affine clause ``2x <= 1`` reaches the numeric mask.
    """

    session = VeritasSession(half_box, vt.MockCharModel(), DecodeConfig())
    session._reset_attempt()
    lo, hi = session._bounds("x")
    interval = Interval(lo, hi)
    assert hi <= 0.5 + 1e-9, "the contractor should intersect 2x <= 1 with [0, 1]"
    checked = 0
    for length in range(1, 5):
        for prefix in itertools.product(CHARSET, repeat=length):
            literal = "".join(prefix)
            if not session._number_prefix_ok(literal, False):
                continue
            if Interval.from_numeric_prefix(literal, interval, False).is_empty():
                continue
            if not session._number_stop_ok(
                literal, interval, False, "x", (False, False), "x", session._declared_bounds("x")
            ):
                continue
            checked += 1
            value = float(literal)
            assert 0.0 <= value <= 1.0
            assert value * 2 <= 1.0 + 1e-9, literal
    assert checked > 20


def test_accepted_prefixes_are_actually_extendable():
    """A masked prefix must be completable; otherwise the mask lies to the model."""

    session = VeritasSession(half_box, vt.MockCharModel(), DecodeConfig())
    session._reset_attempt()
    for length in range(1, 4):
        for prefix in itertools.product(CHARSET, repeat=length):
            literal = "".join(prefix)
            if not session._number_prefix_ok(literal, False):
                continue
            if Interval.from_numeric_prefix(literal, Interval(0.0, 0.5), False).is_empty():
                continue
            # At least one longer or equal literal inside the interval must exist.
            completions = [literal]
            completions += [literal + suffix for suffix in CHARSET]
            assert any(
                session._number_prefix_ok(candidate, False)
                and not Interval.from_numeric_prefix(
                    candidate, Interval(0.0, 0.5), False
                ).is_empty()
                for candidate in completions
            )


@pytest.mark.parametrize("seed", range(8))
def test_random_models_terminate_and_never_emit_unsound_values(seed):
    model = vt.RandomCharModel(seed=seed, stop_logit=1.0)
    session = VeritasSession(ratio_spec, model, DecodeConfig(max_backtracks=6), tokenizer=None)
    result = session.run(model)
    if result.report.status == "PASS":
        assert abs(result.values["a"] + result.values["b"] - 1.0) <= 1e-6
        assert 0.0 <= result.values["a"] <= 1.0
        assert 0.0 <= result.values["b"] <= 1.0
    else:
        # A failure must be explained, never silent.
        assert result.report.summary_line()
        assert result.report.violations or result.report.unknown or result.report.missing
    assert result.telemetry.number_expansions <= DecodeConfig().max_number_nodes * 4


def test_hostile_model_that_always_prefers_zero_still_terminates():
    """The model prefers 0 everywhere; the interior of [0.2, 0.4] is the only solution."""

    class ZeroModel:
        def next_logits(self, prefix):
            scores = {str(d): 10.0 - d for d in range(10)}
            scores["<stop>"] = 9.0
            scores["."] = 1.0
            scores["-"] = 0.0
            return scores

    class Interior(BaseModel):
        x: float

    @vt.verified_schema()
    def interior_spec(p: Interior):
        vt.require(0.2 < p.x)
        vt.require(p.x < 0.4)

    result = vt.generate(model=ZeroModel(), spec=interior_spec)
    assert 0.2 < result.value.x < 0.4
