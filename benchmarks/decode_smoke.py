"""Reproducible decode smoke benchmark (mock model, not a performance claim).

Run:
    python benchmarks/decode_smoke.py --iterations 100

Reports correctness failures and latency percentiles for solver-guided generation using
the deterministic mock model. This measures the reference engine's end-to-end path on the
current machine; it does not measure a real LLM, GPU, or native acceleration.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from pydantic import BaseModel, Field

import veritas as vt
from veritas.testing import MockCharModel


class Linear(BaseModel):
    a: float = Field(ge=0.0, le=1.0)
    b: float = Field(ge=0.0, le=1.0)


@vt.verified_schema()
def linear(p: Linear):
    vt.require(p.a + p.b == 1.0, tol=1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    latencies = []
    failures = 0
    for _ in range(args.iterations):
        start = time.perf_counter()
        try:
            result = vt.generate(model=MockCharModel(), prompt="sum to one", spec=linear)
            ok = result.ok
        except Exception:
            ok = False
        latencies.append((time.perf_counter() - start) * 1000.0)
        failures += 0 if ok else 1

    ordered = sorted(latencies)

    def pct(p):
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    summary = {
        "benchmark": "decode_smoke",
        "model": "veritas.testing.MockCharModel",
        "iterations": args.iterations,
        "failures": failures,
        "latency_ms": {
            "p50": round(pct(50), 3),
            "p95": round(pct(95), 3),
            "max": round(ordered[-1], 3),
            "mean": round(statistics.fmean(latencies), 3),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())