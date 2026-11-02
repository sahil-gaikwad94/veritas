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

This measures the Python reference path only. It does not measure a real LLM, GPU kernels,
KV-cache surgery, or the experimental Rust workspace.