"""Tier-0 ``pattern`` masking: exhaustive agreement with :mod:`re`.

The maskable subset must be *exactly* compatible with ``re.search`` (JSON-Schema
``pattern`` semantics) on both satisfaction and prefix feasibility.  Anything outside the
subset must be rejected loudly instead of being masked with weaker semantics, so these
tests enumerate every string up to a small length over a small alphabet and compare
against the reference implementation.
"""

import itertools
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from veritas.pattern import PatternConstraint, UnsupportedPattern, compiles

ALPHABET = "ab1"
PATTERNS = [
    "abc",
    "^abc$",
    "^[A-Z]{3}$",
    "^(foo|ba[rz])$",
    "a*b",
    "a+b?",
    "^a{1,3}$",
    "^a{2,}$",
    "^(ab|a)+c$",
    "^\\d{2}-\\d{2}$",
    "^[a-z0-9_]+$",
    "^[-+]?[0-9]+(\\.[0-9]+)?$",
    "^(yes|no)$",
    "^[^0-9]+$",
    "^(?:ab){2}$",
    "^x?y?z?$",
    "^a.c$",
    "^[0-9]{1,3}(,[0-9]{3})*$",
]


def _strings(max_len: int):
    for size in range(max_len + 1):
        for letters in itertools.product(ALPHABET, repeat=size):
            yield "".join(letters)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_satisfaction_matches_re(pattern):
    constr = PatternConstraint.compile(pattern)
    for text in _strings(4):
        assert constr.satisfied(text) == (re.search(pattern, text) is not None), (
            pattern,
            text,
        )


@pytest.mark.parametrize("pattern", PATTERNS)
def test_can_stop_matches_satisfied(pattern):
    constr = PatternConstraint.compile(pattern)
    for text in _strings(3):
        assert constr.can_stop(text) == constr.satisfied(text)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_prefix_feasibility_never_prunes_a_witness(pattern):
    """A pruned prefix must have no satisfying extension."""

    constr = PatternConstraint.compile(pattern)
    extensions = list(_strings(4))
    for text in _strings(3):
        if constr.prefix_feasible(text):
            continue
        for suffix in extensions:
            assert not constr.satisfied(text + suffix), (pattern, text, suffix)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_witness_survives_the_incremental_mask(pattern):
    """Every character of a satisfying string must be allowed at its step."""

    constr = PatternConstraint.compile(pattern)
    for text in _strings(4):
        if not constr.satisfied(text):
            continue
        for index, ch in enumerate(text):
            prefix = text[:index]
            assert constr.prefix_feasible(prefix)
            assert ch in constr.allowed_next(prefix), (pattern, text, index)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_allowed_next_is_exhaustive(pattern):
    """No allowed character may lead to an infeasible prefix."""

    constr = PatternConstraint.compile(pattern)
    for text in _strings(3):
        if not constr.prefix_feasible(text):
            continue
        for ch in constr.allowed_next(text):
            assert constr.prefix_feasible(text + ch), (pattern, text, ch)


@pytest.mark.parametrize(
    "pattern",
    ["(?i)abc", "^(a)\\1$", "\\bfoo\\b", "(?P<x>a)(?P=x)"],
)
def test_unsupported_patterns_are_rejected(pattern):
    assert not compiles(pattern)
    with pytest.raises(UnsupportedPattern):
        PatternConstraint.compile(pattern)


def test_compile_is_cached_and_immutable_enough():
    first = PatternConstraint.compile("^[ab]+$")
    second = PatternConstraint.compile("^[ab]+$")
    assert first is second
    assert first.satisfied("abab")
    assert not first.satisfied("abc")


def test_whitespace_classes_are_supported():
    constr = PatternConstraint.compile("^\\s+$")
    assert constr.satisfied(" \t")
    assert not constr.satisfied(" x ")


def test_matches_reference_uses_search_semantics():
    constr = PatternConstraint.compile("[0-9]+")
    # ``re.search`` finds the digits anywhere; an unanchored mask must agree.
    assert constr.satisfied("xx42yy")
    assert not constr.satisfied("")
    assert not constr.satisfied("xxxx")
