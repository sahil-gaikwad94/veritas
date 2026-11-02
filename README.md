# Veritas

**Neuro-symbolic solver-guided decoding and formal verification.**

Veritas turns a Python/Pydantic schema plus a small constraint DSL into a compiled
specification, then uses that specification at every layer of generation: a JSON grammar
PDA masks tokens, interval/affine contractors prune numeric domains, 0-DoF fields are
spliced without sampling, CDCL-style conflict analysis backtracks, and the final instance
is verified concretely before it is returned. Results can be shipped with a self-contained
proof certificate.

This repository contains the **working Python reference engine** and a Rust workspace
whose crates are still scaffolding (see [Status and limits](#status-and-limits)).

---

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest tests/ -q
.venv/bin/veritas --help
```

The shipped engine is pure Python and installs with setuptools; no Rust toolchain is
required. Optional extras are available: `.[z3]` for the SMT oracle, `.[cas]` for
SymPy/NumPy, and `.[rl]` for HuggingFace dataset export. For source-tree development
without installation, set `PYTHONPATH=python`.

### A schema that cannot lie

```python
import veritas as vt
from pydantic import BaseModel, Field


class Portfolio(BaseModel):
    weights: dict
    expected_return: float
    volatility: float
    sharpe_ratio: float


COV = {
    ("AAPL", "AAPL"): 0.048,
    ("AAPL", "MSFT"): 0.031,
    ("AAPL", "NVDA"): 0.042,
    ("MSFT", "MSFT"): 0.039,
    ("MSFT", "NVDA"): 0.036,
    ("NVDA", "NVDA"): 0.095,
}
RET = {"AAPL": 0.14, "MSFT": 0.13, "NVDA": 0.24}


@vt.verified_schema()
def portfolio(p: Portfolio):
    vt.require(vt.keys_subset(p.weights, {"AAPL", "MSFT", "NVDA"}))
    vt.require(vt.sum(p.weights.values()) == 1.0, tol=1e-6)
    vt.require(vt.forall(p.weights.values(), lambda w: (0.05 <= w) & (w <= 0.50)))
    vt.derive(p.expected_return == vt.dot(p.weights, RET), tol=1e-5)
    vt.derive(p.volatility == vt.sqrt(vt.quad_form(p.weights, COV)), tol=1e-5)
    vt.derive(p.sharpe_ratio == (p.expected_return - 0.04) / p.volatility, tol=1e-5)


report = portfolio.verify(
    {"weights": {"AAPL": 0.40, "MSFT": 0.35, "NVDA": 0.25}}
)
assert report.status == "PASS"
print(report.values)          # derived fields were filled
print(report.spec_digest)     # stable specification hash
```

`FAIL` reports include the violated constraints, residuals, a minimal unsat core and
relaxation hints. `INCOMPLETE` (missing or not-yet-evaluable fields) is never treated as a
pass. `report.raise_if_failed()` raises `vt.VerificationError` for guards and tools.

### Solver-guided generation

```python
from veritas.testing import StructuredReasoningModel
import veritas as vt

model = StructuredReasoningModel()      # deterministic mock; any next_logits adapter works
result = vt.generate(model=model, prompt="produce a portfolio", spec=portfolio)

print(result.raw)                       # exact JSON that was decoded
print(result.value)                     # validated model (or values dict)
print(result.report.status)             # verified before the result is returned
```

The engine is model-agnostic: a model exposes `next_logits(prefix_text) -> Mapping[str, float]`.
`veritas.testing` supplies deterministic character models for tests; the bridges package
adapts real HF, vLLM, SGLang and llama.cpp tokenizers.

---

## Command line

```bash
# Compile a schema and run the 0-token feasibility gate.
python -m veritas check-spec path/to/schema.py --json

# Verify a JSON instance; exit 0 only for PASS.
python -m veritas verify path/to/schema.py instance.json

# Re-verify a proof certificate offline; exit 0 only when every check passes.
python -m veritas verify-proof proof.json

# Render a decode trace to a standalone HTML report.
python -m veritas inspect trace.json --out report.html

# Infer a schema module from JSON examples, then compile the generated module.
python -m veritas synth --from-json examples.json --out generated_schema.py
python -m veritas check-spec generated_schema.py
```

`--from-doc` is deliberately explicit about requiring a model adapter rather than
pretending to infer from prose.

---

## Architecture

| Layer | Modules | Responsibility |
|---|---|---|
| Authoring | `decorators.py`, `compiler/` | Compile Pydantic models + `require/derive/prefer/forall` into VIR; dependency graph; container analysis |
| Pre-flight | `compiler/preflight.py`, `analysis.py`, `linear.py` | Interval contraction, AGDecode container feasibility, exact affine feasibility, exact strict-bound refutation, checked string/pattern witnesses; contradictory specs fail before generation |
| Tier 0 grammar | `json_pda.py`, `pda.py`, `pattern.py` | Deterministic JSON pushdown automaton, exact token masks over a vocabulary, canonical rendering, incremental `pattern`/length string masks |
| Tier 1 numerics | `interval.py`, `linear.py`, `crane_mode.py` | Affine forms, HC4 contractor, residual containers, clause simplification |
| Tier 2 search | `nano_solver.py`, `cdcl_jumpback.py` | Deterministic linear solver with unsat cores, CDCL conflict branches, tabu prefixes |
| Acceleration | `zero_dof.py`, `token_healer.py` | Ready-field derivation, canonical JSON Jump-Forward splices, boundary healing |
| Decoding | `decode.py` | Session/engine orchestration, masks, splices, final verification, telemetry and trace |
| Verification | `evaluate.py`, `instance.py` | Total concrete evaluator, tri-state logic, derived fill, reports, minimal cores |
| Proofs | `proof/certificate.py` | Self-contained SMT-LIB2 + witness certificates, hashes, offline verification, tamper detection |
| Domain logic | `cartridges/` | Portfolio math, SQL safety, tool policies, scheduling, statistics, dimensional analysis, SMILES, code guardrails |
| Integrations | `bridges/` | HF/vLLM/SGLang/llama.cpp masking, aborting async streams, Gumbel-max diffusion, OpenAI proxy |
| Ecosystem | `ltlf.py`, `rl/`, `cli/`, `synth.py` | Finite-trace policies, DPO/GRPO exports, commands, schema synthesis |

### What the tiers guarantee

- **Tier 0** rejects any token that would make the JSON prefix unparsable, and allows EOS
  only when the document is complete. The mask is computed against the supplied vocabulary;
  it is not a heuristic character check. String fields are masked by `minLength`/
  `maxLength` *and* by their JSON-Schema `pattern`: `pattern.py` compiles the regex to a
  Thompson NFA and reports exactly which characters keep the prefix extendable. Patterns
  outside the maskable subset (inline flags, backreferences, lookaround, word boundaries)
  are rejected loudly and fall back to length-only masking with an entry in
  `telemetry["pattern_fallbacks"]`, so the final verifier — never the mask — has the last
  word.
- **Tier 1** contracts numeric domains and dynamic-container residual sums. Contradictions
  restore the previous domain snapshot instead of poisoning the search. Contraction is
  outward-rounded, so a feasible equality such as `a + b == 1` can never be reported empty
  because of float error. Open (`gt`/`lt`) bounds are represented by the adjacent
  representable value for reals and by the neighbouring integer for ints, and the excluded
  endpoint is refused at the stop decision as well as in the contractor. Open bounds are
  *also* enforced for container elements (`forall(lst, lambda v: 0 < v < 1)`).
- **Numeric search** is a bounded depth-first walk over digits and `.`/`-`, ordered by model
  logits and pruned by the reachable interval, with a deterministic interior fallback when
  the budget runs out. A greedy literal that leads to a dead end is abandoned for a
  feasible sibling instead of failing the whole attempt.
- **Tier 2** only returns `UNSAT` when it can prove it; budget/timeout yields `UNKNOWN`.
- **Final verification** evaluates every constraint on the concrete object. A generated
  result is returned only if the report is `PASS`; if decoding exhausts its budget, the
  engine emits a canonical valid object rather than truncated JSON. Structural failures
  (missing required field, wrong type, pattern mismatch) are reported as concrete
  `outcomes` naming the offending field, so `VerificationError` says *what* to fix instead
  of "0 satisfied, 0 violated, 0 unevaluated".
- **Soft preferences** (`vt.prefer`) steer generation: after a `PASS`, violated soft bounds
  are intersected into the declared domain and the attempt is repeated while the weighted
  preference score improves. Preferences never override hard constraints, and the accepted
  result is the best-scoring verified one (`telemetry["preference_score"]`).

Equality tolerances are applied at final verification, while contractor cuts use exact
bounds. This avoids the classic false-infeasibility bug where an epsilon-padded cut rejects
a feasible instance.

---

## Proof certificates

```python
certificate = vt.build_certificate(portfolio, {"weights": {"AAPL": 0.4, "MSFT": 0.35, "NVDA": 0.25}})
assert vt.verify_proof(certificate).ok

certificate.save("proof.json")
assert vt.verify_proof("proof.json").ok
```

A certificate embeds the versioned VIR, the witness, an SMT-LIB2 rendering and SHA-256
hashes. Verification is offline and re-evaluates the constraints from the embedded spec.
The SMT export reports its coverage explicitly (`smt2_complete`, `smt2_omitted`); omitted
constraints are still checked by the VIR evaluator.

These hashes are **unkeyed integrity checks, not authentication**. Anyone can recompute
them, so bind verification to a trusted spec with `vt.verify_proof(cert, expected_spec=...)`
or sign the artifact in your application. Arbitrary Python/Pydantic validators are not
portable; certificates for non-portable schemas require the original trusted schema to
re-run those validators.

---

## Runtime policies and tools

`vt.WorldState` models declared tools, preconditions, effects and budgets. Tool calls are
recorded, preconditions are checked before execution, and finite-trace LTLf policies
(`always`, `eventually`, `before`, `responds`, `at_most`, `never`) can reject a trace after
the fact. `vt.verified_tool(spec=..., world_state=...)` verifies a tool's return value before
the caller sees it and can attach a proof.

```python
from veritas.ltlf import always, before
from veritas.cartridges.sec import WorldState

state = WorldState()
state.assert_fact("authenticated")
state.before_tool_call("transfer", {"amount": 10})
```

---

## Domain cartridges

| Cartridge | Concrete checks |
|---|---|
| `quant` | Simplex/concentration limits, covariance PSD, Cholesky, portfolio stats, Black–Scholes call/put, implied volatility by bisection, no-arbitrage bounds, GAAP/cash-flow identities |
| `sql` | Real tokenization, table/column/alias resolution, declared foreign-key joins, required tenant predicates, read-only enforcement, structural warnings for constructs it cannot verify |
| `sec` | Declared tools, preconditions/effects, budgets, snapshots, LTLf policy evaluation |
| `or_sched` | Precedence DAG with cycle detection, earliest-start solver, deadlines/releases, interval-capacity overload detection, critical path |
| `stat` | Probability simplex, Fréchet bounds, Bayes consistency, covariance PSD, d-separation via ancestral moral graphs |
| `phys` | Rational-exponent dimensional algebra, AST equation checking, unit-aware quantities, transcendental-function dimension checks |
| `chem` | SMILES tokenizer/parser, branches, ring closures, charges, implicit hydrogens, valence limits, formula and molecular weight |
| `code` | Scope-aware undefined-identifier detection, literal division by zero, literal index bounds, syntax errors |

These are deterministic reference implementations with no mandatory NumPy/z3/SymPy
dependency. The core DSL and numerical helpers use pure-Python reference paths.

---

## Bridges

- **HuggingFace / vLLM / SGLang / llama.cpp**: build an exact JSON token mask from the
  tokenizer vocabulary and attach it as a logits processor. The processors accept lists,
  NumPy arrays and torch tensors. llama.cpp hosts that only accept a grammar string get a
  generic JSON GBNF; if a spec is supplied, that path fails closed rather than pretending
  the generic grammar enforces the spec.
- **Async stream proxy**: supervises raw text, SSE `data:` frames and OpenAI/Anthropic delta
  objects. It raises on malformed prefixes, incomplete termination and final spec violations,
  with an optional violation callback.
- **Diffusion**: Gumbel-max sampling and confidence-ordered unmasking.
- **OpenAI/Instructor**: `patch_openai(client, spec=...)` injects the VIR payload, requests a
  JSON schema, verifies the returned JSON and raises `VerificationError` on failure.

The bridge tests use deterministic fake tokenizers/clients. They validate the adapter logic,
not a live vendor deployment.

---

## Verification

```bash
PYTHONPATH=python .venv/bin/python -m pytest tests/ -q
```

Current status: **303 tests passing** (compiler, preflight, instance verification, JSON PDA,
token healing, decode engine, pattern masking, numeric search, structural diagnostics,
proofs, cartridges, bridges, LTLf/RL and CLI end-to-end).

Useful focused suites:

```bash
.venv/bin/python -m pytest tests/test_instance_verification.py -q
.venv/bin/python -m pytest tests/test_decode_engine.py -q
.venv/bin/python -m pytest tests/test_pattern.py tests/test_numeric_search.py -q
.venv/bin/python -m pytest tests/test_decode_robustness.py -q
.venv/bin/python -m pytest tests/test_proofs.py -q
.venv/bin/python -m pytest tests/test_bridges.py tests/test_cli_preflight.py -q
```

---

## Status and limits

- The **Python engine is the complete, tested product surface** in this repository.
- The `crates/` Rust workspace is still scaffolding. No Rust toolchain (`cargo`/`rustc`) is
  available in this environment, so nothing in `crates/` was compiled or claimed as working.
- The grammar tier recognizes **JSON**, not arbitrary context-free grammars. Schema-guided
  masking supports a documented JSON Schema subset (objects/arrays/scalars, `anyOf`/`oneOf`,
  local `$ref`, required/arity/range/enum); unsupported keywords fail loudly. Token masking
  is exact over a supplied vocabulary but is a Python reference implementation, not a SIMD
  kernel.
- `require` comparisons are exact by default; pass `tol=` explicitly when an engineering
  tolerance is intended. `derive` keeps a numeric equality tolerance by default.
- The certificate verifier is self-contained. The optional Z3 oracle is exercised when
  `z3-solver` is installed; SymPy/CAS paths remain optional.
- `pattern` masking covers a documented regex subset. Constructs that would need semantic
  case folding or unbounded lookaround (`(?i)`, backreferences, word boundaries, lookaround)
  are rejected and generation degrades to length-only masking with an explicit telemetry
  entry; the verifier then rejects any mismatch instead of silently accepting it. Only the
  first `pattern` on a field drives the mask when several are declared.
- Numeric literals are bounded by `DecodeConfig.max_integer_digits`,
  `max_fraction_digits` and `max_number_nodes`. A schema whose only solutions need more
  digits than that is reported as a dead end (with the culprit field), not silently
  rounded into a violation.
- GPU/runtime bridges are adapter-tested with fakes. They are not integration-tested against
  live HF/vLLM/SGLang/llama.cpp servers, and vendor APIs may need host-specific wiring.
- `--from-doc` requires a model adapter and intentionally fails loudly instead of inventing a
  schema.

The design documents that drove the implementation are in
`Veritas Architecture And Math Spec.md`, `Veritas Api And Feature Handbook.md` and
`Veritas Implementation Blueprint And Prompts.md`.
