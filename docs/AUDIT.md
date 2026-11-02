# Veritas engineering audit

Audit date: 2026-11-01. Baseline: `f536a73`, Python 3.11, 122 passing tests.

## Product intent

The three design volumes describe a common, useful product: author constraints beside a
Pydantic model, compile them once, use them during generation, verify the concrete result,
and carry a portable witness certificate. The intended differentiators are deterministic
derived fields, numeric domain propagation, model-independent adapters, and domain-specific
verification cartridges. Performance targets, native Rust acceleration, KV-cache surgery,
unbiased diffusion sampling, and industrial theorem proving are a roadmap—not properties
established by the baseline implementation.

A trustworthy core must precede those optimizations. In particular, a fast path may return
`UNKNOWN`; it may not turn an unsupported formula into `true`.

## Findings and disposition

| Priority | Finding | Impact | Remediation |
|---|---|---|---|
| P0 | Compiler ignored unknown statements/operators/calls and resolved DSL calls by suffix | A schema could compile to no constraints and verify anything | Strict supported-language compiler, source diagnostics, immutable constant capture, no `eval` |
| P0 | SMT adapters returned `SAT` without asserting/checking a formula | False assurance | Real optional Z3 translation; explicit `UNKNOWN`/`UNAVAILABLE`; timeout and core handling |
| P0 | Linear solver used lossy coefficient cutoffs, checked difference and linear constraints separately, ignored unsupported input, and extracted incorrect deletion cores | False SAT/UNSAT and misleading explanations | Exact rational feasibility, unified constraints, resource limits, witnessed SAT, correct core deletion |
| P0 | Preflight translated exhausted/unsupported checks into `SAT` | False feasibility guarantee | Three-state gate; only a checked witness or complete decision may return SAT |
| P0 | Pydantic validation happened after DSL PASS; normalized values were not rechecked; strict bounds, nullability, aliases/extras and custom validators were not represented faithfully | Invalid object could be returned or certified | Validate before/after derivation, verify normalized data, richer types, portable schema validation, explicit custom-validator boundary |
| P0 | JSON PDA accepted leading zeros and malformed exponent delimiters; parser accepted NaN/duplicate keys | Grammar/proof boundary bypass | State-machine number grammar, Unicode support, strict bounded JSON parser, duplicate rejection |
| P0 | Generation returned a `VerifiedResult` on FAIL; lists were emitted as objects; unsupported types became strings | Broken product contract | Fail-closed public generation, typed containers, explicit unsupported-schema errors, diagnostic session remains available |
| P0 | HF adapter used raw BPE strings, one batch prefix for every row, and generic JSON rather than the supplied schema | Wrong masks and cross-request corruption | Tokenizer-aware vocabulary, per-row state/prompt offsets, JSON Schema prefix supervision |
| P1 | Certificates omitted model semantics and silently omitted SMT constraints; hashes were described as authentication | Overclaimed portability/tamper resistance | Embed structural schema, validate it offline, report SMT coverage, bind rendered script, document unkeyed hashes |
| P1 | Token healing fallback claimed a valid prefill without checking text preservation | KV/text divergence | Explicit exactness flag and fail-closed exact API; full-prefix retokenization fallback |
| P1 | Tool guards did not enforce temporal policies/budgets atomically, did not commit successful effects, and returned unnormalized output | Agent safety bypass and state corruption | Prospective policy checks, transactional commit/rollback, argument binding, validated return values, async support |
| P1 | SQL tenant isolation was a substring check; multiple statements and unsupported syntax could pass | Security-critical false assurance | Single-statement read-only subset, identifier validation, conjunctive per-table tenant predicates, explicit unsupported errors |
| P1 | RL exported empty preference pairs; writes were not atomic; decode conflicts lacked prompts/repairs | Poisoned training data and unreproducible exports | Complete-pair filtering, actual decode provenance, atomic JSONL/manifest writes |
| P1 | Packaging required an unavailable Rust toolchain although the shipped engine was Python | Installation failure | Pure-Python wheel/editable install; native workspace explicitly experimental |
| P2 | Rust modules were placeholders, including a solver returning true | Dangerous if exposed | No native acceleration claim; safe unavailable status and native roadmap |
| P2 | Domain helpers had mathematical edge cases (PSD ridge, discounted expiry, missing returns, resource-search completeness, chemistry valence) | Incorrect domain results | Focused fixes and adversarial tests; conservative coverage reporting |

## Trust boundaries

* A Veritas certificate is a **recheckable witness artifact**, not a theorem-prover proof or
  a digital signature. Anyone can recompute an unkeyed SHA-256 hash. Bind verification to a
  trusted spec digest, or authenticate the artifact in the application's signing system.
* Offline verification covers the embedded VIR and supported structural JSON Schema. Arbitrary
  Python validators cannot be serialized safely and require the original trusted schema.
* White-box masking is exact only with respect to the supplied tokenizer vocabulary and the
  recognizer's documented schema subset. Vendor engines must explicitly install the processor;
  attaching an unused Python attribute is not an integration.
* Heuristic search cuts are not CDCL conflict lemmas. A bounded failed search does not prove
  global UNSAT. Only the final concrete verifier determines whether a result is accepted.

## Validation strategy

Regression tests reproduce each P0 defect before the fix. The suite additionally includes
randomized grammar round trips, rational-solver cross-checks, schema validation, batch
isolation, proof tampering, tool rollback, and clean-wheel installation tests. Benchmarks must
report hardware, sample counts, correctness failures, and latency distributions; no speedup is
claimed from a mock model.
## Remediation status (2026-11-02)

The P0/P1 trust-boundary findings above have been remediated in the Python reference
engine and are covered by the regression suite. Additional hardening completed in this
pass:

* `require` comparisons are exact by default (explicit `tol=` opt-in); `derive` retains a
  numeric equality tolerance. Present JSON `null` is distinct from a missing field.
* VIR deserialization rejects malformed versions, variable tables, expression nodes,
  tolerances, tagged values and duplicate mapping keys instead of silently defaulting.
* Schema-guided masking keeps `anyOf`/`oneOf` candidates independent, validates completed
  containers against the original schema, resolves local `$ref`, and rejects recursive or
  unsupported schema keywords.
* Certificates expose and bind SMT coverage (`smt2_complete`, `smt2_omitted`); `seal()`
  recomputes unkeyed hashes, and trust-anchor verification uses `expected_spec`.
* Interval HC4 narrowing for negative-coefficient inequalities is sound; PSD testing no
  longer launders indefinite matrices through a ridge; scheduling critical paths include
  release times; unknown-element molecular weight fails closed.
* `verify_json` reports grammar errors as `InstanceFormatError`; llama.cpp grammar-only
  hosts fail closed when a spec cannot be enforced by the generic JSON grammar.
* Code-synth guardrails handle walrus bindings, match captures, function defaults and
  annotations, and emit warnings for global/nonlocal bindings. Schema synthesis sanitizes
  identifiers and emits deterministic key universes.
* Packaging is pure-Python setuptools with a `py.typed` marker; wheel and editable installs
  are smoke-tested. CI runs ruff, pytest, and wheel build on Python 3.11/3.12.

Validation: `ruff check python tests` clean; `pytest tests/ -q` → **159 passed**; wheel
build and isolated import/CLI smoke pass; `benchmarks/decode_smoke.py` reports 0 failures.

### Remaining known limits

* The Rust/Wasm/PyO3 workspace is experimental scaffolding and was not compiled here.
* Schema masking supports the documented subset only; unsupported keywords fail loudly.
* Generic llama.cpp GBNF is JSON-only; schema-specific GBNF generation is not implemented.
* Quadratic interval contraction is not implemented; concrete verification remains
  authoritative for quadratic constraints.
* Certificates are recheckable witness artifacts, not signed proofs; non-portable Python
  validators require the original trusted schema.


## Second pass: empirical probing (2026-11-02, branch `arena/167f6e2b-veritas`)

The remediation above was validated against the *claims* in the docs. This pass instead
ran adversarial probes against the shipped engine and fixed what they exposed. Every item
below is reproduced by a regression test in `tests/`.

| Symptom found by probing | Root cause | Fix |
|---|---|---|
| A structural failure produced a report with **no** outcomes, so the error text was `INCOMPLETE: 0 satisfied, 0 violated, 0 unevaluated` | `verify_values` returned early on JSON-Schema errors without synthesising outcomes | Schema errors are flattened into per-field outcomes (`missing` = unevaluated, type/pattern/bounds = violated); `VerificationError` appends the concrete reasons |
| `Field(gt=0.0)` (with `a + b == 1`) generated `a = 1`, violating the open bound | Contractor domains were closed, and the stop check compared against the outward-rounded interval rather than the declared type bound | Open bounds become the adjacent representable value (real) / neighbouring integer (int); the stop check uses declared bounds; committing an assignment that leaves an open-bounded field with no interior value is a conflict naming the *upstream* culprit |
| `a >= 1` and `a < 1` passed preflight as `UNKNOWN` (and `gt`/`le` pairs too) | `simplify_clause` closes strict inequalities for the rational solver, so the pair looked satisfiable at the boundary | Exact 1-D bound refutation (strict-aware, integer-aware) is added to preflight; infeasibility now raises `InfeasibleSpecError` |
| `a + b == 1` with `a = 0.023` made the contractor report a *false* dead end | HC4 linear/sum narrowing did not round outwards, so `1 - 0.977` fell one ULP outside `[0.023, 0.023]` | Outward rounding on the residual and quotient in every linear contraction, plus a ULP-slack retry when committing a point assignment; regression test pins `a=0.023 ⇒ b≈0.977` |
| `forall(lst, lambda v: 0.2 < v < 0.4)` emitted `0.4` | Element analysis discarded strictness, and container slots had no declared-range check | `ContainerInfo` carries element exclusivity; the mask receives the analysed range as its declared range |
| A greedy numeric literal that dead-ends (`0.4` when `v < 0.4`) failed the whole attempt | `_emit_number` was a single forward pass with no local backtracking | Bounded DFS over digits ordered by model logits, plus a deterministic interior fallback when the budget is exhausted; per-field `number_expansions` telemetry |
| `vt.prefer` was verified but never influenced generation | No repair loop existed (the handbook documents one) | MaxSMT-style repair: violated soft bounds are intersected into the declared domain, the attempt is repeated while the weighted score improves, and the best verified result is kept (`preference_score`, `preference_repairs`) |
| `reasoning_mode="crane_cot"` returned `reasoning=""` | The flag existed but nothing produced a scratchpad | CRANE scratchpad is produced before decoding, kept out of the JSON span, and exposed as `VerifiedResult.reasoning` (mined equations included) |
| Token-healed splices were documented but never planned | `TokenHealer` was only used by tests | Zero-DoF/derived fields record a splice plan (`backup_tokens`, `token_ids`, `exact`, `offset`) in `result.splices` and in telemetry |
| `domain_cuts` recorded a cut for a field that was never cut; `backtrack_fields` stayed empty | The cut path recorded whatever the last field was, and never labelled the field | Cuts are attributed from the violated clause's variables with per-field spans/entropies/conflict counts, and record the culprit in `backtrack_fields` |
| `Field(pattern=...)` was ignored during masking | `VirType` had no `pattern` field and there was no regex engine | New `pattern.py` (Thompson NFA from `re._parser`) provides exact prefix feasibility; `pattern`/`format` now round-trip through VIR and serde |
| `check-spec` returned `UNKNOWN` for any schema with a pattern-constrained string, even when a witness obviously existed | The witness builder filled strings with `"x" * min_len`, which cannot satisfy a pattern | Pre-flight asks the recognizer for a shortest accepted string inside the declared *and* schema-level length window, so pattern schemas now get a checked `SAT` witness |

Verified state after this pass: `pytest tests/ -q` → **303 passed**; `ruff check python
tests benchmarks` → clean; `benchmarks/decode_smoke.py` → 0 failures on the mock model.

### Limits specific to this pass

* The pattern subset is deliberately narrow: inline flags, backreferences, lookaround and
  word boundaries are rejected rather than approximated. Rejected patterns degrade to
  length-only masking with a `pattern_fallbacks` entry, and the verifier rejects a
  mismatch. Only the first declared `pattern` per field drives the mask.
* Numeric literals are bounded by `max_integer_digits`, `max_fraction_digits` and
  `max_number_nodes`; a solution needing more digits than that is reported as a dead end
  rather than rounded into (or out of) a violation.
* The repair loop maximises the *weighted fraction* of satisfied soft preferences. It is a
  greedy tightening loop, not an optimal MaxSMT procedure, and it stops as soon as a round
  does not improve the score.
* Outward rounding means contracted bounds are wide by up to a few ULPs. Decode decisions
  use those bounds, the verifier uses exact values, and the verifier remains authoritative.

### CI dependency regression (PR #3)

The first CI run exposed an unused NumPy import in the Cholesky compiler test and a
matrix-input `vt.quad_form` path that silently required NumPy. The test now checks the
pure-Python factor directly (`L Lᵀ = Σ`), matrix quadratic forms use a dependency-free
path, and a regression forces NumPy imports to fail. The core CI environment deliberately
installs only `dev,z3` so this contract remains tested.
