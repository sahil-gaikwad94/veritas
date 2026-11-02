# Benchmarks

Veritas benchmarks must report hardware, sample count, correctness failures, and latency
distributions. A mock-model smoke run is a reproducibility check, **not** a speedup claim.

## Decode smoke

```bash
python benchmarks/decode_smoke.py --iterations 100
```

The script runs `vt.generate` against the deterministic `MockCharModel` and a two-variable
simplex schema, then prints JSON with failure count and p50/p95/max/mean latency. It exits
nonzero if any generation fails verification.

Example run in the audit environment (30 iterations, Python 3.11.2,
`Linux-6.1.158+-x86_64-with-glibc2.36`):

```json
{
  "benchmark": "decode_smoke",
  "model": "veritas.testing.MockCharModel",
  "iterations": 30,
  "failures": 0,
  "latency_ms": { "p50": 1.022, "p95": 1.289, "max": 1.512, "mean": 1.063 }
}
```

Re-run after the robustness pass (100 iterations, same environment as recorded by the
script's own `python`/`platform` fields):

```json
{
  "benchmark": "decode_smoke",
  "model": "veritas.testing.MockCharModel",
  "iterations": 100,
  "failures": 0,
  "latency_ms": { "p50": 1.053, "p95": 1.193, "max": 2.279, "mean": 1.087 },
  "python": "3.11.2",
  "platform": "Linux-6.1.158+-x86_64-with-glibc2.36"
}
```

The numeric search, pattern masking and preference repair added in that pass are not on
this schema's hot path, so the smoke numbers stay in the same ~1 ms range; they are a
reproducibility check, not a performance claim. Per-field work is visible in telemetry
(`number_expansions`, `string_mask_prunes`, `contractor_prunes`, `preference_repairs`)
rather than in this aggregate.

This measures the Python reference path only. It does not measure a real LLM, GPU kernels or
KV-cache surgery. For the native engine, see the A/B below.

## Native engine: measured, not claimed

`benchmarks/native_speedup.py` runs **the same Python call sites twice**, once with the
compiled extension and once with `VERITAS_NO_NATIVE=1`, and reports median/p95 per call for
each. Nothing is re-implemented for the benchmark: the workloads are `TokenMasker.valid_mask`
(605-token JSON vocabulary, cache cleared so each call does real work) and
`Interval.from_numeric_prefix` (160 literals spanning `0.3`, `00`, `1e-3`, `-7.5`). Each side
runs in its own subprocess so import-time state cannot leak between them.

```console
$ maturin build --release -m crates/veritas-py/Cargo.toml -o dist
$ pip install --no-deps dist/*.whl
$ python benchmarks/native_speedup.py            # table
$ python benchmarks/native_speedup.py --json     # machine-readable
```

Without the extension the script prints the import error and exits 0 rather than inventing a
comparison.

Measured by the `native extension` CI job on a GitHub `ubuntu-latest` runner (Python 3.12.15,
run 41395204893, 2026-11-02), against the wheel that job had just built:

| workload (per call) | pure Python | native | speedup |
| --- | ---: | ---: | ---: |
| `TokenMasker.valid_mask`, 605-token vocabulary | 4067 µs | 227 µs | **17.9x** |
| `Interval.from_numeric_prefix`, 160 literals | 5.69 µs | 3.88 µs | **1.5x** |

The shape is what the design predicts. `valid_mask` is a per-token loop — 605 clones and steps
of the automaton, each with Python object overhead — so moving it into Rust removes almost all
of the cost. A single numeric prefix is a handful of float operations, so what the native path
saves there is mostly the interpreter's work, and a third of the remaining time is the
Python↔Rust call boundary itself. That is also why the delegation is worth keeping despite the
modest second number: it is never *slower*, and it becomes the dominant win in the loop where
it matters.

These are wall-clock medians on shared runners and vary by tens of percent between runs. They
answer "is the accelerator wired up and worth its boundary cost", not "how fast is Veritas".

Two honest caveats: the per-call Python↔Rust boundary costs a fraction of a microsecond, so
cheap calls (short numeric prefixes) may gain little or nothing from delegation — the win is in
per-token loops; and these figures are wall-clock medians on shared CI runners, which vary by
tens of percent between runs. The numbers answer "is the accelerator wired up and worth its
boundary cost", not "how fast is Veritas".
