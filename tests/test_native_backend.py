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


def test_kill_switch_forces_the_pure_path(monkeypatch):
    monkeypatch.setenv(native.DISABLE_ENV, "1")
    assert native.disabled_by_env()
    assert not native.available()
    assert native.DISABLE_ENV in (native.load_error() or "")
    assert native.info()["native_available"] is False
    assert native.info()["load_error"] == f"disabled by {native.DISABLE_ENV}"
    assert native.wire_version() is None
    # The engine keeps working: the reference path answers instead.
    assert native.prefix_status("{\"a\": 1}") == "complete"
    assert native.mask("{\"a\": ", VOCABULARY, EOS) == TokenMasker(
        VOCABULARY, eos_token=EOS
    ).valid_mask("{\"a\": ")
    assert not native.numeric_prefix("0.3", Interval(0.0, 1.0)).is_empty()
    assert native.solve_linear([]) is None
    assert native.contract({0: (0.0, 1.0)}, []) is None


def test_kill_switch_is_read_per_call(monkeypatch):
    # Not a module-level constant: a benchmark flips it without re-importing the package.
    monkeypatch.delenv(native.DISABLE_ENV, raising=False)
    before = native.available()
    monkeypatch.setenv(native.DISABLE_ENV, "1")
    assert native.available() is False
    monkeypatch.setenv(native.DISABLE_ENV, "0")
    assert native.available() is before


def _load_benchmark():
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "native_speedup_benchmark", root / "benchmarks" / "native_speedup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speedup_benchmark_only_times_legal_prefixes():
    module = _load_benchmark()
    vocabulary = module._vocabulary()
    assert len(vocabulary) > 500
    for prefix in module._prefixes(vocabulary):
        machine = JSONPushdownAutomaton()
        assert machine.feed(prefix) in (True, False)  # never raises
    # measure() filters to the prefixes a decoder can actually hold.
    report = module.measure(1, 1)
    assert report["native_available"] is native.available()
    assert report["vocabulary"] == len(vocabulary)
    for workload in ("mask", "numeric_prefix"):
        assert report[workload]["p50_us"] >= 0.0
        assert report[workload]["p95_us"] >= report[workload]["p50_us"]


# --------------------------------------------------------------------------------------
# Randomized differential checks. The corpus above is hand-picked; these sweep the input
# spaces the corpus cannot enumerate, so "the accelerator agrees with the reference" is a
# property rather than a list. They need the extension: without it there is nothing to
# compare, and the shim would be compared with itself.
# --------------------------------------------------------------------------------------

DIFFERENTIAL_SEED = 0x5EED_0F_7A11


def _differential_rng():
    import random

    return random.Random(DIFFERENTIAL_SEED)


def _reference_prefix_status(text: str) -> str:
    machine = JSONPushdownAutomaton()
    if not machine.feed(text):
        return "invalid"
    probe = machine.clone()
    probe.finish()
    return "complete" if probe.accepts() else "incomplete"


JSON_ALPHABET = "{}[]\":,01.-e+tru falsn\\\n\t5"


def _random_prefix(rng, length: int) -> str:
    return "".join(rng.choice(JSON_ALPHABET) for _ in range(length))


def _random_document(rng, depth: int = 0) -> str:
    """A small *valid* document. Prefixes are generated from these, not from noise: a decoder
    only ever holds prefixes of documents the grammar admits."""
    kind = rng.choice(["scalar"] if depth > 2 else ["scalar", "array", "object"])
    if kind == "scalar":
        return rng.choice(["null", "true", "false", "0", "-1.5", "1e3", '"a"', '"x y"', "7"])
    if kind == "array":
        items = [_random_document(rng, depth + 1) for _ in range(rng.randint(0, 3))]
        return "[" + ", ".join(items) + "]"
    keys = ["a", "b", "c"][: rng.randint(1, 3)]
    return "{" + ", ".join(f'"{key}": {_random_document(rng, depth + 1)}' for key in keys) + "}"


def _random_case(rng) -> str:
    """A complete document, a truncation of one, or a one-character mutation of one."""
    document = _random_document(rng)
    roll = rng.random()
    if roll < 0.35:
        return document
    if roll < 0.85:
        text = document[: rng.randint(0, len(document) - 1)]
        return text + (" " if rng.random() < 0.5 else "")
    position = rng.randrange(len(document))
    return document[:position] + rng.choice(JSON_ALPHABET) + document[position + 1 :]


def _random_vocabulary(rng) -> list[str]:
    tokens: list[str] = []
    for _ in range(rng.randint(6, 18)):
        if rng.random() < 0.15:
            tokens.append("")  # an empty token must never be allowed
        elif rng.random() < 0.5:
            tokens.append(_random_prefix(rng, rng.randint(1, 3)))
        else:
            tokens.append(rng.choice(["}", "]", ", ", ": ", '"}', '"]', "null", "true", "0", "1.5", "e3", "-"]))
    return tokens


@requires_native
def test_random_prefixes_agree_with_the_reference_automaton():
    extension = pytest.importorskip("veritas_py")
    rng = _differential_rng()
    verdicts = {"complete": 0, "incomplete": 0, "invalid": 0}
    for _ in range(600):
        text = _random_case(rng)
        expected = _reference_prefix_status(text)
        verdicts[expected] += 1
        assert extension.prefix_status(text) == expected, f"prefix {text!r}"
    # A sweep that only ever produces one verdict proves nothing about the other two.
    assert all(count > 20 for count in verdicts.values()), verdicts


@requires_native
def test_random_masks_agree_with_the_reference_masker():
    rng = _differential_rng()
    checked = 0
    for _ in range(120):
        vocabulary = _random_vocabulary(rng)
        prefix = _random_case(rng)
        if _reference_prefix_status(prefix) == "invalid":
            continue
        native_mask = native.mask_or_none(prefix, vocabulary, None)
        assert native_mask is not None, "the native mask must answer for a legal prefix"
        assert native_mask == TokenMasker(vocabulary).valid_mask(prefix), (
            f"prefix {prefix!r} vocabulary {vocabulary!r}"
        )
        checked += 1
    assert checked > 40, checked


def _same_interval(lo, hi, other_lo, other_hi) -> bool:
    both_empty = lo > hi and other_lo > other_hi
    if both_empty:
        return True
    if (lo > hi) != (other_lo > other_hi):
        return False
    return math.isclose(lo, other_lo, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(
        hi, other_hi, rel_tol=1e-12, abs_tol=1e-12
    )


@requires_native
def test_random_numeric_prefixes_agree_with_the_reference():
    extension = pytest.importorskip("veritas_py")
    rng = _differential_rng()
    non_empty = 0
    for _ in range(500):
        digits = rng.randint(0, 3)
        literal = rng.choice(["", "-"]) + str(rng.randint(0, 999))
        if digits:
            literal += "." + "".join(str(rng.randint(0, 9)) for _ in range(digits))
        if rng.random() < 0.2:
            literal += "e" + str(rng.randint(0, 4))
        center = float(literal)
        width = rng.choice([0.5, 2.0, 20.0, 200.0])
        bounds = Interval(center - width * rng.uniform(0.1, 0.9), center + width * rng.uniform(0.1, 0.9))
        for allow_exp in (False, True):
            lo, hi = extension.numeric_prefix(literal, bounds.lo, bounds.hi, allow_exp)
            reference = Interval._from_numeric_prefix_python(literal, bounds, allow_exp)
            assert _same_interval(lo, hi, reference.lo, reference.hi), (
                f"literal {literal!r} bounds {bounds} allow_exp={allow_exp}: "
                f"native [{lo}, {hi}] vs python [{reference.lo}, {reference.hi}]"
            )
            if not reference.is_empty():
                non_empty += 1
    assert non_empty > 200, non_empty


# --------------------------------------------------------------------------------------
# A crash in the accelerator must cost speed, not the decode. PyO3 reports a Rust panic as
# `pyo3_runtime.PanicException`, which is a `BaseException`, so the fallback cannot be
# `except Exception` — and Ctrl-C still has to work.
# --------------------------------------------------------------------------------------


class _ExtensionThatRaises:
    WIRE_VERSION = 1

    def __init__(self, error):
        self.error = error

    def prefix_status(self, text):
        raise self.error

    def canonical_document(self, text):
        raise self.error

    def allowed_tokens(self, prefix, vocabulary, eos):
        raise self.error

    def numeric_prefix(self, text, lo, hi, allow_exp):
        raise self.error

    def contract(self, domains, constraints, rounds):
        raise self.error

    def solve_linear(self, constraints):
        raise self.error

    def engine_info(self):
        raise self.error


class _RustPanic(BaseException):
    """Stands in for `pyo3_runtime.PanicException`, which derives from `BaseException`."""


def test_a_native_crash_degrades_to_the_reference_path(monkeypatch):
    monkeypatch.delenv(native.DISABLE_ENV, raising=False)
    monkeypatch.setattr(native, "_NATIVE", _ExtensionThatRaises(_RustPanic("panicked at src/pda.rs")))
    assert native.available()
    assert native.prefix_status('{"a": 1}') == "complete"
    assert native.canonical_document('{"b": 2, "a": 1}') == '{"b": 2, "a": 1}'
    assert native.mask('{"a": ', VOCABULARY, EOS) == TokenMasker(
        VOCABULARY, eos_token=EOS
    ).valid_mask('{"a": ')
    span = native.numeric_prefix("0.3", Interval(0.0, 1.0))
    assert span.lo <= 0.3 <= span.hi
    assert native.contract({0: (0.0, 1.0)}, []) is None
    assert native.solve_linear([]) is None
    assert native.info()["name"] == "veritas_py"


def test_the_fallback_still_lets_ctrl_c_through(monkeypatch):
    monkeypatch.delenv(native.DISABLE_ENV, raising=False)
    monkeypatch.setattr(native, "_NATIVE", _ExtensionThatRaises(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        native.prefix_status('{"a": 1}')
    with pytest.raises(KeyboardInterrupt):
        native.mask('{"a": ', VOCABULARY, EOS)
