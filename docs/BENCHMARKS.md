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

This measures the Python reference path only. It does not measure a real LLM, GPU kernels,
KV-cache surgery, or the experimental Rust workspace.