# The native engine: contract, not folklore

Veritas ships a complete pure-Python engine, plus an optional Rust accelerator that answers a
subset of the same questions faster. This document is the contract between the two: what the
accelerator is, when it is used, what happens when it fails, and how the "it cannot change a
verdict" claim is checked.

If you only remember one line: **the Python engine decides; the native engine only answers
faster, and is reverted at the first sign of trouble.**

## What exists

| crate | role |
| --- | --- |
| `crates/veritas-core` | every layer of the reference engine ported to Rust. No dependencies: no crates, no C toolchain, no build scripts |
| `crates/veritas-py` | PyO3 bindings (module `veritas_py`, `abi3-py311`), built with `maturin` |
| `crates/veritas-wasm` | `wasm-bindgen` bindings over the same handles, built for `wasm32-unknown-unknown` |

Installing Veritas never requires a Rust toolchain. The extension is an accelerator a user
opts into:

```console
$ maturin build --release -m crates/veritas-py/Cargo.toml -o dist
$ pip install --no-deps dist/*.whl     # a separate distribution: it provides veritas_py only
```

`pip install --no-deps` matters: the wheel provides the `veritas_py` module and nothing else,
while `python/veritas` stays the package you import. Installing the wheel *over* an editable
install makes pip treat one as a rename of the other and uninstall it.

## Selection and failure behaviour

`veritas.native` resolves the engine per call, never at import time:

| situation | `available()` | behaviour |
| --- | --- | --- |
| extension missing or unimportable | `False` | pure Python; `load_error()` names the exception |
| `WIRE_VERSION` mismatch | `False` | pure Python; a wire change is a hard boundary, not a guess |
| `VERITAS_NO_NATIVE=1` (or `true`/`yes`/`on`; `0`/`false`/`no`/`off` are off) | `False` | pure Python; `load_error()` is `disabled by VERITAS_NO_NATIVE` |
| extension raises `Exception` | `True` | that one call falls back; the error is swallowed |
| extension panics (PyO3's `PanicException`, a `BaseException`) | `True` | that one call falls back |
| Ctrl-C / `sys.exit` inside a native call | — | propagates; never converted into a retry |

The panic row is the subtle one. `except Exception` would not catch a Rust panic, because
`pyo3_runtime.PanicException` derives from `BaseException`. Every native call site therefore
catches `BaseException` and re-raises `KeyboardInterrupt`/`SystemExit`; the rest of the
exception hierarchy is treated as "the accelerator is broken, use the reference path".

`veritas.doctor()` and `veritas.native.info()` report the live engine and the reason it is
missing, which is the fastest way to answer "why is this slow".

## What is actually accelerated

Two call sites, both on the decode hot path:

* `TokenMasker.valid_mask` — the per-token loop a decoder runs on every step. This is where
  the interpreter cost lives (one automaton clone and step per vocabulary entry).
* `Interval.from_numeric_prefix` — the numeric mask while a number is being typed.

Exposed but **not** wired in: `native.contract` (HC4 revision) and `native.solve_linear`
(exact feasibility). They are parity-tested and available to callers, but the decode engine
uses its own stateful Python contractor, and the stateless wire call would cost more than it
saves on the small systems the decoder contracts. Saying so is more useful than pretending the
whole engine changed language.

## Wire protocol

`WIRE_VERSION = 1`. Both bindings speak the same JSON shapes, and a version mismatch disables
the accelerator rather than guessing:

* `prefix_status(text) -> "complete" | "incomplete" | "invalid"`
* `canonical_document(text) -> str | None`
* `allowed_tokens(prefix, vocabulary, eos) -> list[int]`, `mask(...) -> list[bool]`
* `numeric_prefix(text, lo, hi, allow_exp) -> (lo, hi)`, empty intervals as `(+inf, -inf)`
* `contract(domains_json, constraints_json, rounds) -> domains_json`. JSON cannot print ±∞, so on
  the wire `null` is an unbounded endpoint and `[]` is an empty domain; both directions
  translate, so Python callers only ever see floats (an empty domain arrives as
  `[inf, -inf]`). A round-trip property in `soundness.rs` pins the mapping bit for bit,
  including the intervals that are not numbers.
* `solve_linear(constraints_json) -> {"status": "sat"|"unsat"|"unknown", ...}`
* `engine_info() -> {"name", "version", "wire_version", "abi"}`

## How the parity claim is checked

Four independent layers, all in CI:

1. **Rust unit tests** — 67 examples over the modules, run by `cargo test -p veritas-core`.
2. **Rust properties** — 9 tests in `crates/veritas-core/tests/soundness.rs`: contractor
   refutations never contain a satisfying point, UNSAT cores are re-solved standalone, the
   numeric prefix mask keeps every legal completion, the automaton and the parser agree on
   random strings, healing never reports an unverified splice, rendering and parsing are inverse.
3. **Hand-picked parity corpus** — 38 cases in `tests/test_native_backend.py` comparing the
   native answers with the pure-Python ones.
4. **Randomized differential sweeps** (same file, `requires_native`): 600 prefix verdicts over
   documents, their truncations and mutations; 120 random vocabularies against
   `TokenMasker.valid_mask`; 500 numeric literals in both `allow_exp` branches. The sweeps also
   assert that the native path *answered* rather than silently falling back, and that the
   verdict distribution covers all three outcomes.

`cargo clippy --all-targets -D warnings` and both binding builds (pyo3, wasm) run in the same
`rust` job, so a module that stops compiling cannot be committed unnoticed.

## What it is worth

Measured by `benchmarks/native_speedup.py`, which runs the same Python call sites twice — once
with the extension, once with `VERITAS_NO_NATIVE=1` — on a GitHub `ubuntu-latest` runner
(Python 3.12.15):

| workload (per call) | pure | native | speedup |
| --- | ---: | ---: | ---: |
| `TokenMasker.valid_mask`, 605-token vocabulary | 4067 µs | 227 µs | **17.9x** |
| `Interval.from_numeric_prefix`, 160 literals | 5.69 µs | 3.88 µs | **1.5x** |

The per-token loop is where Rust pays for itself; a single numeric prefix is a few float
operations, so what remains is mostly the Python↔Rust boundary. See `docs/BENCHMARKS.md` for
provenance and caveats, and run the script yourself before quoting either number.
