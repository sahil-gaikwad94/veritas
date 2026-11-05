"""What does the native engine actually buy? Measure it, don't claim it.

The workload is the *same Python code* twice: ``TokenMasker.valid_mask`` and
``Interval.from_numeric_prefix`` are the real call sites the decoder uses, and the only thing
that changes between the two runs is ``VEXX_NO_NATIVE``. That makes the comparison honest:
no re-implementation, no re-warmed caches, no cherry-picked API.

Run:
    python benchmarks/native_speedup.py            # comparison table
    python benchmarks/native_speedup.py --json     # machine-readable
    VEXX_NO_NATIVE=1 python benchmarks/native_speedup.py --worker   # one side only

Without the compiled extension the script says so and exits 0: a benchmark that cannot measure
must not pretend to.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vocabulary() -> list[str]:
    """A JSON-shaped token vocabulary: keys, numbers, punctuation, literals."""
    tokens: list[str] = []
    for index in range(60):
        tokens.append(f'"field_{index:03d}": ')
        tokens.append(f'{{"field_{index:03d}": ')
        tokens.append(f'{{"field_{index:03d}": {{')
    tokens.extend(
        [
            "}",
            "},",
            "}]",
            "}]}",
            "[",
            "]",
            ", ",
            ": ",
            "null",
            "true",
            "false",
            '"',
            "0",
            "1",
            "2",
            "3",
            "5",
            "7",
            "0.",
            "0.5",
            "0.25",
            "1.0",
            "1e-3",
            "-",
            "\n",
        ]
    )
    for index in range(200):
        tokens.append(f"{index}")
        tokens.append(f"0.{index:03d}")
    return tokens


def _prefixes(vocabulary: list[str]) -> list[str]:
    """Progressively longer partial documents, the way a decoder actually advances."""
    prefixes = [
        "",
        "{",
        '{"field_000": ',
        '{"field_000": 1',
        '{"field_000": 1, ',
        '{"field_000": 1, "field_001": ',
        '{"field_000": 1, "field_001": 2}',
        '{"field_000": {"field_001": [',
        '{"field_000": {"field_001": [1, 2',
        '{"field_000": {"field_001": [1, 2]}',
        '{"field_000": {"field_001": [1, 2]}, "field_002": ',
        '{"field_017": -',
        '{"field_017": 0.',
        '{"field_017": 0.25',
        '[',
        "[[",
        "[[1], [2",
        '{"a": "x',
        '{"a": "x\\',
    ]
    for token in vocabulary[:8]:
        prefixes.append('{"field_000": ' + token)
    return prefixes


def _literals() -> list[str]:
    literals: list[str] = []
    for index in range(40):
        literals.append(f"{index}")
        literals.append(f"0.{index:02d}")
        literals.append(f"-{index}.{index % 10}")
        literals.append(f"{index}.5")
    return literals


def measure(iterations: int, repeats: int) -> dict:
    from vexx import native
    from vexx.interval import Interval
    from vexx.json_pda import JSONPushdownAutomaton, TokenMasker

    vocabulary = _vocabulary()
    # A decoder only ever holds prefixes the grammar allowed, so an invalid one is not a
    # workload — drop it instead of benchmarking an exception path.
    prefixes = [
        prefix
        for prefix in _prefixes(vocabulary)
        if JSONPushdownAutomaton().feed(prefix)
    ]
    literals = _literals()
    bounds = Interval(0.0, 1.0)
    masker = TokenMasker(vocabulary)

    def mask_block() -> None:
        for prefix in prefixes:
            masker.clear_cache()
            masker.valid_mask(prefix)

    def prefix_block() -> None:
        for literal in literals:
            Interval.from_numeric_prefix(literal, bounds)

    def time_block(block, units: int) -> dict:
        for _ in range(max(1, repeat_warmup(iterations))):
            block()
        samples: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            for _ in range(repeats):
                block()
            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed / (repeats * units) / 1000.0)  # microseconds per call
        samples.sort()
        return {
            "p50_us": round(statistics.median(samples), 3),
            "p95_us": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
            "calls_per_s": int(1e6 / samples[len(samples) // 2]) if samples[len(samples) // 2] else 0,
        }

    return {
        "mode": "native" if native.available() else "pure",
        "native_available": native.available(),
        "reason": native.load_error(),
        "python": platform.python_version(),
        "vocabulary": len(vocabulary),
        "mask": time_block(mask_block, len(prefixes)),
        "numeric_prefix": time_block(prefix_block, len(literals)),
    }


def repeat_warmup(iterations: int) -> int:
    return max(1, iterations // 10)


def _worker(args) -> int:
    print(json.dumps(measure(args.iterations, args.repeats)))
    return 0


def _child(iterations: int, repeats: int, disabled: bool) -> dict:
    env = dict(os.environ)
    if disabled:
        env["VEXX_NO_NATIVE"] = "1"
    else:
        env.pop("VEXX_NO_NATIVE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(REPO_ROOT, "python"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--iterations",
        str(iterations),
        "--repeats",
        str(repeats),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(f"worker failed ({result.returncode}):\n{result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")

    native_side = _child(args.iterations, args.repeats, disabled=False)
    pure_side = _child(args.iterations, args.repeats, disabled=True)

    report = {
        "python": native_side["python"],
        "vocabulary": native_side["vocabulary"],
        "native_available": native_side["native_available"],
        "reason": native_side["reason"],
        "workloads": {},
    }
    for workload in ("mask", "numeric_prefix"):
        pure = pure_side[workload]
        native = native_side[workload]
        speedup = pure["p50_us"] / native["p50_us"] if native["p50_us"] else None
        report["workloads"][workload] = {
            "pure": pure,
            "native": native,
            "speedup": round(speedup, 2) if speedup else None,
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    if not native_side["native_available"]:
        print("native extension not available:", native_side["reason"])
        print("build it with: maturin build --release -m crates/vexx-py/Cargo.toml -o dist")
        print("               pip install --no-deps dist/*.whl")
        print("the pure-Python engine is unaffected; nothing to compare.")
        return 0

    print(f"python {report['python']}, vocabulary {report['vocabulary']} tokens")
    print(f"{'workload':<18}{'pure µs/call':>14}{'native µs/call':>16}{'speedup':>10}")
    for workload, numbers in report["workloads"].items():
        print(
            f"{workload:<18}{numbers['pure']['p50_us']:>14.3f}"
            f"{numbers['native']['p50_us']:>16.3f}{str(numbers['speedup']) + 'x':>10}"
        )
    print("\np50 over repeated blocks of the same call sites; VEXX_NO_NATIVE pins the pure path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
