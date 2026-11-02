"""The native engine is an accelerator, never a second source of truth.

These tests run in two modes:

* without the compiled extension (the default wheel) they prove the shim falls back and that
  the reported reason is honest, and the parity assertions are skipped;
* with ``veritas_py`` importable (the wheels job) they prove the native answers agree with the
  pure-Python ones on a corpus, so preferring the native path cannot change any verdict.
"""

from __future__ import annotations

import math

import pytest

from veritas import native
from veritas.interval import Interval
from veritas.json_pda import JSONPushdownAutomaton, JSONSyntaxError, TokenMasker

PREFIXES = [
    "",
    " ",
    "{",
    "{}",
    "{\"a\"",
    "{\"a\":",
    "{\"a\": ",
    "{\"a\": 1",
    "{\"a\": 1}",
    "{\"a\": 1, \"b\": [2, 3]}",
    "{\"a\" 1}",
    "[1,]",
    "[1",
    "[[1], [2, [3]]]",
    "nul",
    "null",
    "nullx",
    "\"a\\",
    "\"a\\u00",
    "0.",
    "01",
    "-",
    "1e",
    "1e+",
    "1e+2",
    "abc",
    "tru",
    "true",
]

# The end-of-sequence slot must stay an empty token: the engine refuses to let EOS collide
# with text that could be sampled.
VOCABULARY = [
    "{\"a\": ",
    "}",
    "\"a\"",
    ": ",
    "1",
    "null",
    "]",
    "x",
    "",
]
EOS = VOCABULARY.index("")

requires_native = pytest.mark.skipif(
    not native.available(),
    reason=f"veritas_py not built ({native.load_error()})",
)


def python_prefix_status(text: str) -> str:
    machine = JSONPushdownAutomaton()
    if not machine.feed(text):
        return "invalid"
    probe = machine.clone()
    probe.finish()
    return "complete" if probe.accepts() else "incomplete"


def test_shim_reports_whether_the_extension_loaded():
    info = native.info()
    assert info["native_available"] is native.available()
    if native.available():
        assert info["name"] == "veritas_py"
        assert info["load_error"] is None
        assert native.wire_version() == 1
    else:
        assert info["name"] == "python"
        assert info["load_error"]


@pytest.mark.parametrize("prefix", PREFIXES)
def test_fallback_prefix_status_matches_the_automaton(prefix):
    assert native.prefix_status(prefix) == python_prefix_status(prefix)


def test_fallback_mask_matches_the_engine():
    for prefix in ["", "{\"a\": ", "{\"a\": 1", "{\"a\": 1}", "[1"]:
        assert native.mask(prefix, VOCABULARY, EOS) == TokenMasker(
            VOCABULARY, eos_token=EOS
        ).valid_mask(prefix)


def test_fallback_numeric_prefix_matches_the_engine():
    bounds = Interval(0.0, 1.0)
    for prefix in ["", "0", "0.", "0.3", "0.25,", "1", "-", "abc"]:
        span = native.numeric_prefix(prefix, bounds)
        expected = Interval._from_numeric_prefix_python(prefix, bounds)
        assert span.lo == expected.lo and span.hi == expected.hi


def test_unavailable_engine_returns_none_instead_of_guessing():
    if native.available():
        assert native.contract({0: (0.0, 1.0)}, []) is not None
    else:
        assert native.contract({0: (0.0, 1.0)}, []) is None
        assert native.solve_linear([]) is None


@requires_native
@pytest.mark.parametrize("prefix", PREFIXES)
def test_native_prefix_parity(prefix):
    assert native.prefix_status(prefix) == python_prefix_status(prefix)


@requires_native
@pytest.mark.parametrize("prefix", ["", "{\"a\": ", "{\"a\": 1", "{\"a\": 1}", "[1"])
def test_native_mask_parity(prefix):
    assert native.mask_or_none(prefix, VOCABULARY, EOS) == TokenMasker(
        VOCABULARY, eos_token=EOS
    ).valid_mask(prefix)


@requires_native
def test_native_mask_rejects_an_invalid_prefix():
    # The Python engine raises, so the shim must refuse to answer rather than mask.
    assert native.mask_or_none("{\"a\" 1}", VOCABULARY, None) is None
    with pytest.raises(JSONSyntaxError):
        TokenMasker(VOCABULARY).valid_mask("{\"a\" 1}")


@requires_native
def test_native_numeric_prefix_never_loses_a_value():
    bounds = Interval(0.0, 1.0)
    for prefix in ["0", "0.", "0.3", "0.25,", "1", "-"]:
        native_span = native.numeric_prefix_or_none(prefix, bounds)
        python_span = Interval._from_numeric_prefix_python(prefix, bounds)
        assert native_span is not None
        if python_span.is_empty():
            assert native_span.is_empty()
            continue
        assert not native_span.is_empty()
        # Outward rounding may differ by an ULP, but no reachable value may be dropped.
        assert native_span.lo <= python_span.lo + 1e-12
        assert native_span.hi >= python_span.hi - 1e-12


@requires_native
def test_native_contractor_refutes_what_python_refutes():
    domains = {0: (0.0, 1.0), 1: (0.0, 1.0)}
    impossible = [{"kind": "sum", "vars": [0, 1], "rhs": 5.0, "op": "eq"}]
    answer = native.contract(domains, impossible)
    assert answer["status"] == "empty"
    feasible = [{"kind": "sum", "vars": [0, 1], "rhs": 1.0, "op": "eq"}]
    answer = native.contract(domains, feasible)
    assert answer["status"] == "feasible"
    assert answer["unsupported"] == 0


@requires_native
def test_native_solver_witnesses_are_real():
    constraints = [
        {"coeffs": {"0": 1.0, "1": 1.0}, "rhs": 1.0, "op": "le"},
        {"coeffs": {"0": 1.0}, "rhs": 0.3, "op": "ge"},
        {"coeffs": {"1": 1.0}, "rhs": 0.3, "op": "ge"},
    ]
    answer = native.solve_linear(constraints)
    assert answer["status"] == "sat"
    model = answer["model"]
    assert model["0"] + model["1"] <= 1.0 + 1e-9
    assert model["0"] >= 0.3 - 1e-9 and model["1"] >= 0.3 - 1e-9

    unsat = native.solve_linear(
        [
            {"coeffs": {"0": 1.0}, "rhs": 0.6, "op": "ge"},
            {"coeffs": {"0": 1.0}, "rhs": 0.5, "op": "le"},
        ]
    )
    assert unsat["status"] == "unsat"
    assert unsat["core"] == [0, 1]


@requires_native
def test_native_document_rendering_is_canonical():
    assert native.canonical_document('{"a":  1}') == '{"a": 1}'
    assert native.canonical_document('{"a": 1, "a": 2}') is None
    assert native.canonical_document("NaN") is None


def test_numeric_prefix_fallback_is_still_sound():
    # Even without the extension, the shim must not invent a narrower span.
    bounds = Interval(0.05, 0.5)
    span = native.numeric_prefix("0.3", bounds)
    assert not span.is_empty()
    assert math.isclose(span.lo, 0.3, abs_tol=1e-12)
    assert span.hi <= 0.5


def test_doctor_reports_the_loaded_engine():
    import veritas

    report = veritas.doctor()
    assert report["version"] == veritas.__version__
    assert report["python"] == ("accelerated" if native.available() else "pure")
    assert report["engine"]["native_available"] is native.available()
    if not native.available():
        assert report["engine"]["load_error"]
    assert veritas.native is native
