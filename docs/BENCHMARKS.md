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
comparison. Pure-path baselines from one Linux x86-64 dev box (Python 3.11): `valid_mask`
≈ **2.9 ms** per call on that 605-token vocabulary — the pure path clones and steps the
automaton once per token — and `from_numeric_prefix` ≈ **4.8 µs** per call. The native side of
those two numbers is published automatically by the `native extension` CI job, which runs this
script against the wheel it just built; the job's log is the source of truth because a
benchmark recorded on one developer machine is not evidence about another.

Two honest caveats: the per-call Python↔Rust boundary costs a fraction of a microsecond, so
cheap calls (short numeric prefixes) may gain little or nothing from delegation — the win is in
per-token loops; and these figures are wall-clock medians on shared CI runners, which vary by
tens of percent between runs. The numbers answer "is the accelerator wired up and worth its
boundary cost", not "how fast is Veritas".
