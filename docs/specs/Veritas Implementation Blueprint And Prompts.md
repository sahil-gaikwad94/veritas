# `veritas` — Neuro-Symbolic Solver-Guided Decoding & Formal Verification Engine
## Volume III: Zero-Cost Implementation Blueprint, Workspace Layout & Master Coding Agent Prompts (`v2.0.0`)

**Document Status:** Engineering Execution Blueprint (`v2.0.0` — Upgraded from `v1.0.0-alpha`)  
**Target Workflow:** Engineered for rapid, zero-cost execution using open-source toolchains (Rust `cargo`, `PyO3`/`maturin`, `uv`, `tree-sitter`, `pydantic` v2, `sympy`, `pytest`, `hypothesis`, optional `z3-sys`) and AI coding agents (Claude Code, Cursor, Windsurf, Cline).

## 1. Upgraded Monorepo Architecture & Directory Tree (`v2.0.0`)

`veritas` `v2.0` is structured as a hybrid **Rust + Python + WebAssembly workspace** managed by `maturin` and `uv`. Critically, `crates/veritas-core` now includes `nano_solver.rs` (a zero-dependency pure-Rust Dual Simplex + Difference Logic solver) alongside `smt_bridge.rs` (optional `z3-sys` feature flag), enabling 4MB prebuilt `abi3` wheels that install in `<2 seconds` via `pip install veritas-ai`.

```text
veritas/
├── Cargo.toml                          # Rust workspace manifest (veritas-core, veritas-py, veritas-wasm)
├── pyproject.toml                      # Maturin + uv build config (default = pure-Rust; extras = [z3, cas, rl])
├── README.md
├── crates/
│   ├── veritas-core/                   # Core Rust execution engine (zero C++ deps by default)
│   │   ├── Cargo.toml                  # Optional feature: `z3-backend`
│   │   └── src/
│   │       ├── lib.rs                  # Module exports & unified DecoderStep API
│   │       ├── vir.rs                  # Veritas Intermediate Representation (VIR) AST, ResidualAccumulators & SoftPrefs
│   │       ├── pda.rs                  # Tier-0 Pushdown Automaton & SIMD AVX-512/NEON token bitmasking
│   │       ├── crane_mode.rs           # CRANE dual-mode <think>/<emit> controller & passive equation miner
│   │       ├── shadow_ast.rs           # Incremental Tree-sitter CST parser & clause boundary detector
│   │       ├── interval.rs             # Tier-1 IEEE-754 HC4-Revise + Affine Arithmetic (AF2) + Volume Twist
│   │       ├── nano_solver.rs          # Tier-2A Pure-Rust Incremental Dual Simplex (QF_LRA) & Bellman-Ford (QF_IDL)
│   │       ├── smt_bridge.rs           # Tier-2B Incremental Z3/cvc5 pool, push/pop DAG, MaxSMT & UnsatCore
│   │       ├── zero_dof.rs             # Deterministic 0-DoF evaluator & cascaded expression scheduler
│   │       ├── token_healer.rs         # BPE boundary token healing & Jump-Forward prefill splice builder
│   │       ├── cdcl_jumpback.rs        # Entropy-weighted UnsatCore culprit attribution, Tabu trie & Domain Cuts
│   │       ├── diffusion.rs            # Gumbel-Max DP span chart & bidirectional factor-graph island verifier
│   │       └── kv_hook.rs              # PagedAttention / RadixAttention block-table rewind & Jump-Forward extend
│   ├── veritas-py/                     # PyO3 zero-copy Python bindings (abi3-py311)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                  # PyVeritasEngine, PyDecodeSession, DLPack in-place logit mask
│   └── veritas-wasm/                   # WebAssembly build for browser-side proof verification & trace-viewer UI
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # Wasm verify_certificate() & interactive trace replay
├── python/
│   └── veritas/
│       ├── __init__.py                 # Public API: @verified_schema, require, derive, prefer, forall, generate
│       ├── decorators.py               # @verified_schema, @verified_tool & AST inspection entrypoints
│       ├── compiler/
│       │   ├── __init__.py
│       │   ├── ast_transpiler.py       # Python AST -> VIR + SymPy polynomial canonicalization
│       │   ├── schema_lowering.py      # Pydantic v2 model -> Topological JSON CFG + SymbolicVariableTable
│       │   ├── dependency_graph.py     # Bipartite variable-constraint DAG & 0-DoF topological key sorter
│       │   └── preflight.py            # 0-Token static SAT gate & Slack-LP minimum relaxation diagnostics
│       ├── bridges/
│       │   ├── __init__.py
│       │   ├── hf_transformers.py      # HuggingFace LogitsProcessor + Jump-Forward KV prefill loop
│       │   ├── vllm_bridge.py          # vLLM V1 LogitsProcessor + BlockManager checkpoint rewind
│       │   ├── sglang_bridge.py        # SGLang RadixAttention compressed FSM + Jump-Forward backend
│       │   ├── llamacpp_bridge.py      # llama-cpp-python custom sampler chain
│       │   ├── instructor_bridge.py    # 1-line drop-in mode for Instructor & OpenAI response_format
│       │   ├── diffusion_bridge.py     # LLaDA / Dream Gumbel-Max factor-graph unmasking scheduler
│       │   └── async_stream_proxy.py   # Black-box SSE supervisor, abort-on-UNSAT & prefix-cached restart
│       ├── proof/
│       │   ├── __init__.py
│       │   └── certificate.py          # VerifiedResult[T], SMT-LIB2 exporter, Lean4 stub & SHA-256 witness
│       ├── rl/
│       │   ├── __init__.py
│       │   └── collector.py            # Step-level DPO / Process-GRPO trajectory collector from CDCL jumpbacks
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py                 # CLI: `veritas check-spec`, `veritas inspect`, `veritas synth`, `veritas verify`
│       └── cartridges/
│           ├── quant.py                # Portfolio simplex, Cholesky quad_form, Black-Scholes, GAAP accounting
│           ├── sql.py                  # Catalog-aware Text-to-SQL FK/alias/column & RLS verifier
│           ├── sec.py                  # LTLf multi-turn agent tool-call safety & budget conservation
│           ├── chem.py                 # SMILES valence automaton, ring parity, Hückel aromaticity
│           ├── or_sched.py             # Difference logic scheduling, AllDifferent, capacity routing
│           ├── phys.py                 # SI dimensional group (Q^7) & Schwartz-Zippel polynomial identity check
│           ├── code_synth.py           # Scope/type-safe code synthesis constraints (AGDecode LFP)
│           └── stat.py                 # Simplex probabilities, Fréchet bounds & DAG d-separation
├── benchmarks/
│   └── veritas_bench/                  # Reproducible benchmark suite vs Outlines, XGrammar, llguidance, Instructor
│       ├── run_jsonschemabench.py
│       ├── run_quant_portfolio.py
│       └── run_gsm_symbolic_crane.py
└── tests/
    ├── test_interval_af2_contractor.py # Property-based Hypothesis tests for HC4 + AF2 + prefix bounds
    ├── test_nano_solver.py             # Unit tests for pure-Rust Dual Simplex & Difference Logic
    ├── test_ast_to_vir_toposort.py     # Compiler tests for topological key sorting & 0-token pre-flight gate
    ├── test_zero_dof_jump_forward.py   # Verify BPE token-healed Jump-Forward splicing on dependent fields
    ├── test_cdcl_domain_cut.py         # Verify UnsatCore attribution, KV-cache rewind & half-space cuts
    └── test_end_to_end_cartridges.py   # Full integration tests across quant, sql, phys, and or_sched
```

## 2. Core Data Structures & Production Rust Implementations (`crates/veritas-core`)

### 2.1 Upgraded Veritas Intermediate Representation (`vir.rs`)

```rust
use serde::{Deserialize, Serialize};

pub type VarId = u32;
pub type ClauseId = u32;
pub type TokenId = u32;
pub type NoiseId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirType {
    Real { min: Option<f64>, max: Option<f64>, tol: f64 },
    Int { min: Option<i64>, max: Option<i64> },
    Bool,
    BitVec { width: u32 },
    FiniteSet { universe: Vec<String> },
    DynamicMap {
        allowed_keys: Option<Vec<String>>,
        min_len: usize,
        max_len: usize,
        val_min: f64,
        val_max: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VirExpr {
    Var(VarId),
    ConstReal(f64),
    ConstInt(i64),
    ConstBool(bool),
    Add(Vec<VirExpr>),
    Sub(Box<VirExpr>, Box<VirExpr>),
    Mul(Vec<VirExpr>),
    Div(Box<VirExpr>, Box<VirExpr>),
    Pow(Box<VirExpr>, i32),
    Sqrt(Box<VirExpr>),
    /// Cholesky-factored PSD quadratic form: w^T Sigma w = || L^T w ||_2^2
    QuadFormCholesky {
        weights_vars: Vec<VarId>,
        cholesky_l: Vec<Vec<f64>>,
    },
    /// Residual sum accumulator over dynamic collections (AGDecode derivative)
    ResidualSum {
        container_id: VarId,
        target_sum: f64,
        elem_min: f64,
        elem_max: f64,
        min_remaining: usize,
        max_remaining: usize,
    },
    Compare { op: CmpOp, lhs: Box<VirExpr>, rhs: Box<VirExpr>, tol: f64 },
    And(Vec<VirExpr>),
    Or(Vec<VirExpr>),
    Implies(Box<VirExpr>, Box<VirExpr>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftPreference {
    pub id: ClauseId,
    pub tag: String,
    pub weight: f64,
    pub expr: VirExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroDofRule {
    pub target_var: VarId,
    pub target_json_key: String,
    pub input_vars: Vec<VarId>,
    pub rhs_expr: VirExpr,
    pub precision_digits: usize,
}
```

### 2.2 Complete Production Implementation of `Interval::from_numeric_prefix` & `AffineForm` (`interval.rs`)

In `v1.0.0-alpha`, `from_numeric_prefix` was left as a `todo!()`. Below is the complete, zero-allocation $O(1)$ implementation for `v2.0.0`, handling negative signs, partial decimals, closed delimiters, and Affine Arithmetic (`AF2`) noise symbols:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const ENTIRE: Self = Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
    pub const EMPTY: Self = Self { lo: f64::INFINITY, hi: f64::NEG_INFINITY };

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi || self.lo.is_nan() || self.hi.is_nan()
    }

    #[inline(always)]
    pub fn width(&self) -> f64 {
        if self.is_empty() { 0.0 } else { (self.hi - self.lo).max(0.0) }
    }

    #[inline(always)]
    pub fn intersect(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// Computes the exact interval of real values reachable by extending `raw_candidate`
    /// as a standard JSON decimal literal (without scientific notation if `allow_exp == false`),
    /// intersected with `domain_bounds`. Runs in < 80 nanoseconds with zero heap allocations.
    pub fn from_numeric_prefix(
        raw_candidate: &str,
        domain_bounds: Self,
        allow_exp: bool,
    ) -> Self {
        let s = raw_candidate.trim_start();
        if s.is_empty() {
            return domain_bounds;
        }

        // Check if the candidate token closes the numeric literal (e.g. ends with ',', '}', ']', ' ')
        let (num_part, is_closed) = match s.find(|c: char| matches!(c, ',' | '}' | ']' | ' ' | '\n' | '\t')) {
            Some(idx) => (&s[..idx], true),
            None => (s, false),
        };

        // Bare sign prefix "-" or "+" before any digits are emitted
        if num_part == "-" {
            return domain_bounds.intersect(Self { lo: f64::NEG_INFINITY, hi: 0.0 });
        }
        if num_part == "+" {
            return domain_bounds.intersect(Self { lo: 0.0, hi: f64::INFINITY });
        }

        let is_neg = num_part.starts_with('-');
        let clean = num_part.strip_suffix('.').unwrap_or(num_part);
        let Ok(base_val) = clean.parse::<f64>() else {
            return Self::EMPTY;
        };

        if is_closed {
            // Exact point interval when closed by a delimiter
            return domain_bounds.intersect(Self { lo: base_val, hi: base_val });
        }

        if allow_exp && (num_part.contains('e') || num_part.contains('E')) {
            return domain_bounds;
        }

        // Case A: Already contains a decimal point (e.g. "0.3" or "-0.35")
        // Any valid continuation without exponent only appends fractional digits in [0, 9]:
        // For positive "0.3" (1 frac digit), completions lie in [0.3, 0.3 + 10^-1].
        if let Some(dot_idx) = num_part.find('.') {
            let frac_digits = (num_part.len() - dot_idx - 1) as i32;
            let ulp = 10.0_f64.powi(-frac_digits);
            let span = if is_neg {
                Self { lo: base_val - ulp, hi: base_val }
            } else {
                Self { lo: base_val, hi: base_val + ulp }
            };
            return domain_bounds.intersect(span);
        }

        // Case B: Leading integer digits without decimal point yet (e.g. "0" or "12")
        // In JSON, leading "0" cannot be followed by more integer digits ("01" is invalid JSON),
        // so "0" can only become 0 or "0.xxx" in [0.0, 1.0) (or (-1.0, 0.0] if "-0").
        if num_part == "0" || num_part == "-0" {
            let span = if is_neg {
                Self { lo: -1.0, hi: 0.0 }
            } else {
                Self { lo: 0.0, hi: 1.0 }
            };
            return domain_bounds.intersect(span);
        }

        // Non-zero integer prefix `k` (e.g. "12"): can either close/add decimal in [k, k+1)
        // OR append more integer digits [10^m * k, 10^m * (k + 1)) for m >= 1.
        // We check if ANY power-of-10 scale intersects `domain_bounds`.
        let mut union_lo = f64::INFINITY;
        let mut union_hi = f64::NEG_INFINITY;
        let abs_v = base_val.abs();
        let mut scale = 1.0_f64;
        for _ in 0..16 {
            let scaled_lo = if is_neg { -(abs_v + 1.0) * scale } else { abs_v * scale };
            let scaled_hi = if is_neg { -abs_v * scale } else { (abs_v + 1.0) * scale };
            let sub = domain_bounds.intersect(Self { lo: scaled_lo, hi: scaled_hi });
            if !sub.is_empty() {
                union_lo = union_lo.min(sub.lo);
                union_hi = union_hi.max(sub.hi);
            }
            scale *= 10.0;
            if !is_neg && abs_v * scale > domain_bounds.hi {
                break;
            }
            if is_neg && -(abs_v * scale) < domain_bounds.lo {
                break;
            }
        }
        Self { lo: union_lo, hi: union_hi }
    }
}

/// Affine Arithmetic (AF2) form: x_hat = center + sum_i (coeff_i * eps_i) + rad_pos * eps_+ - rad_neg * eps_-
#[derive(Debug, Clone)]
pub struct AffineForm {
    pub center: f64,
    pub terms: Vec<(NoiseId, f64)>,
    pub err_radius: f64,
}

impl AffineForm {
    #[inline]
    pub fn to_interval(&self) -> Interval {
        let l1: f64 = self.terms.iter().map(|(_, c)| c.abs()).sum::<f64>() + self.err_radius;
        Interval {
            lo: self.center - l1,
            hi: self.center + l1,
        }
    }
}
```

## 3. Phase-by-Phase Execution Roadmap (Zero-Cost 10-Day Build Plan)

### Phase 1: Workspace Scaffolding, Topological `VIR` Compiler & Pre-Flight Gate (`Days 1–2`)
1. Initialize the hybrid Rust/Python/Wasm workspace using `maturin init --bindings pyo3`.
2. Implement `python/veritas/compiler/` (`ast_transpiler.py`, `schema_lowering.py`, `dependency_graph.py`, `preflight.py`):
   * Parse `@vt.verified_schema` AST + Pydantic v2 models into `VirSpec`.
   * Topologically sort JSON schema fields so free variables precede `vt.derive(...)` 0-DoF variables.
   * Run the **0-Token Pre-Flight Feasibility Check** (plus Slack-LP minimum relaxation diagnostics on `UNSAT`).

### Phase 2: Tier-0 Byte PDA, CRANE Mode Controller & Tier-1 `AF2 + HC4` Contractor (`Days 3–4`)
1. Implement `pda.rs` and `crane_mode.rs` to handle `<think>...</think>` unconstrained scratchpad windows and context-independent SIMD token bitmasks.
2. Implement `interval.rs` with outward-rounded IEEE-754 `HC4-Revise`, `AffineForm` (`AF2`) noise-symbol cancellation, Cholesky quadratic bounds, `ResidualSum` dynamic container lookahead, and the **Future-Volume Twist** $\gamma_t^{\text{Vol}}(v)$.
3. Integrate `tree-sitter-json`, `tree-sitter-python`, and `tree-sitter-sql` in `shadow_ast.rs`.

### Phase 3: `VeritasNano` Pure-Rust Solver, Z3 Bridge & CDCL Domain-Cut Jumpback (`Days 5–7`)
1. Implement `nano_solver.rs`: Pure-Rust incremental Dual Simplex (`QF_LRA`) and Bellman-Ford Difference Logic (`QF_IDL`) with tracked assumption literals and minimal `UnsatCore` extraction in `<150 us`.
2. Implement `smt_bridge.rs` (behind optional `z3-backend` Cargo feature) for non-linear/quantified theories and Weighted MaxSMT (`vt.prefer`).
3. Implement `cdcl_jumpback.rs` + `zero_dof.rs` + `token_healer.rs`:
   * Entropy-weighted culprit selection, Paged/Radix KV-cache block truncation, Tabu prefix trie, and monotonic **Half-Space Domain Cuts**.
   * **Jump-Forward BPE Token-Healed Splicer** that evaluates cascaded 0-DoF variables and returns a single batched prefill token slice.

### Phase 4: Ecosystem Bridges, 8 Cartridges, Wasm Inspector & `VeritasBench` (`Days 8–10`)
1. Implement `vllm_bridge.py`, `sglang_bridge.py`, `hf_transformers.py`, `instructor_bridge.py`, and `async_stream_proxy.py`.
2. Implement `veritas.rl.collector` (exporting CDCL conflict branches into HuggingFace `datasets` Step-DPO / Process-GRPO format) and `veritas-wasm` (`veritas inspect` local visualizer).
3. Implement the 8 domain cartridges (`quant`, `sql`, `sec`, `chem`, `or_sched`, `phys`, `code_synth`, `stat`) and run `benchmarks/veritas_bench/`.

## 4. Six Copy-Pasteable Master Prompts for AI Coding Agents

Paste these self-contained prompts directly into Claude Code, Cursor, Windsurf, or Cline to build each module cleanly with zero hallucinated dependencies.

### Prompt 1: Build the Topological AST-to-VIR Compiler & 0-Token Pre-Flight Gate
```text
You are a principal compiler engineer building `veritas` v2.0, a neuro-symbolic solver-guided decoding engine in Python 3.11+ and Rust.
Implement the front-end compiler in `python/veritas/`:
1. `python/veritas/decorators.py`:
   - Implement `@verified_schema(solver="auto", reasoning_mode="crane_cot", backtrack="cdcl_domain_cut", max_backtracks=8, zero_dof_collapse=True, preflight_check=True)`.
   - Implement DSL primitives: `require(expr, tol=1e-6)`, `derive(expr, tol=1e-6)`, `prefer(expr, weight=1.0, tag="")`, `forall(iterable, predicate)`, `sum(...)`, `dot(w, r)`, `quad_form(w, cov)`, `sqrt(x)`, and `keys_subset(d, allowed_set)`.
2. `python/veritas/compiler/dependency_graph.py`:
   - Build a bipartite DAG between Pydantic schema fields (`VarId`) and `vt.derive(...)` equality rules.
   - Perform a topological sort of the Pydantic model's JSON schema keys so that independent (free) variables are always emitted BEFORE their downstream 0-DoF derived variables. Raise a clear `CircularDerivationError` at decoration time if a cycle exists in `vt.derive`.
3. `python/veritas/compiler/ast_transpiler.py`:
   - Lower `require`, `derive`, and `prefer` AST nodes into our `VirSpec` structure.
   - For `quad_form(w, cov)`, verify that `cov` is positive semi-definite and precompute its lower-triangular Cholesky factor `L` via `numpy.linalg.cholesky` so the Rust contractor can bound `||L^T w||_2^2` without interval dependency blowup.
   - Lower reductions over dynamic dicts/lists into `VirExpr::ResidualSum` descriptors.
4. `python/veritas/compiler/preflight.py`:
   - Before any LLM call, execute a 0-Token Pre-Flight Feasibility Check on base frame `F_0`. If `F_0` is UNSAT, solve a slack-variable linear relaxation to identify the minimal conflicting user constraints and raise `InfeasibleSpecError` with exact parameter relaxation suggestions.
Write comprehensive pytest unit tests in `tests/test_ast_to_vir_toposort.py`.
```

### Prompt 2: Build the Rust `HC4 + AF2` Affine Contractor & Future-Volume Twist Masker
```text
You are a principal Rust systems engineer building `crates/veritas-core/src/interval.rs`.
Implement a zero-allocation (<35 microseconds) Hybrid Interval (`HC4-Revise`) + Affine Arithmetic (`AF2`) Contractor and subword numeric prefix masker:
1. Implement `struct Interval { pub lo: f64, pub hi: f64 }` with outward-rounded IEEE-754 operations (`add`, `sub`, `mul`, `div`, `sqr`, `sqrt`, `intersect`) and `Interval::from_numeric_prefix(raw_candidate: &str, domain_bounds: Self, allow_exp: bool) -> Self`.
2. Implement `struct AffineForm` tracking first-order noise symbols `(NoiseId, f64)` so linear combinations (`w1 + w2 + w3 == 1.0` or `x - x`) cancel shared noise symbols without interval dependency over-approximation.
3. Implement `Hc4AffineContractor`:
   - Support `VirExpr::QuadFormCholesky` using the precomputed Cholesky factor `L` and `VirExpr::ResidualSum` for open container lookahead.
   - Execute bottom-up evaluation + top-down narrowing until fixed-point or until any domain becomes `Interval::is_empty()` (returning `Err(ConflictCore)`).
4. Implement `compute_numeric_token_twist(var_id: VarId, candidate_interval: Interval, beta: f64) -> f64`:
   - Estimate the log-volume ratio of the remaining free variables' contracted box after intersecting `domains[var_id]` with `candidate_interval`, providing O(1) SHIM-style bias correction against Locally Constrained Decoding myopia.
Write property-based tests in Rust verifying that for `w1 + w2 + w3 == 1.0` with `0.05 <= wi <= 0.50`, setting `w1 = 0.10` narrows `w2` to `[0.40, 0.50]` and immediately rejects prefix `"0.3"` for `w2`.
```

### Prompt 3: Build `VeritasNano` — The Pure-Rust Incremental Dual Simplex & Difference Logic Solver
```text
You are an SMT solver engineer building `crates/veritas-core/src/nano_solver.rs`.
To ensure `pip install veritas-ai` works with zero C++ dependencies in a 4MB wheel, implement `VeritasNanoSolver` in pure Rust:
1. Implement an incremental **Dual Simplex (`QF_LRA`)** tableau solver over rational/f64 bounds with `push()`, `pop()`, and `assert_and_track(row_constraint, tracker_id: ClauseId)`:
   - Maintain basic and non-basic variables with incremental bound updates (`x_i >= l_i`, `x_i <= u_i`).
   - When a bound tightening violates feasibility, extract the minimal **UnsatCore (`Vec<ClauseId>`)** directly from the Dual Farkas certificate (the non-basic variables with non-zero multipliers in the infeasible tableau row)!
2. Implement an incremental **Difference Logic (`QF_IDL` / `QF_RDL`)** graph solver for constraints of the form `x_i - x_j <= c` (used by `veritas.or` scheduling):
   - Detect negative-weight cycles incrementally via incremental Bellman-Ford / Tarjan subtree disassembly and return the cycle edges as the minimal `UnsatCore`.
3. Expose a unified `Tier2Oracle` trait implemented by both `VeritasNanoSolver` (default fast path, <150 us) and `Z3SmtOracle` (optional feature `z3-backend` for `QF_NRA` / `QF_BV` / quantifiers).
Write unit tests in `tests/test_nano_solver.py` verifying exact Farkas UnsatCore extraction and push/pop state restoration.
```

### Prompt 4: Build the Entropy-Weighted CDCL Jumpback & Half-Space Domain-Cut Engine
```text
You are a formal verification and systems engineer building `crates/veritas-core/src/cdcl_jumpback.rs`.
1. Maintain a chronological stack of `ClauseCheckpoint` records containing `clause_id`, `var_id`, `committed_val: f64`, `start_token_idx`, `end_token_idx`, `mean_shannon_entropy`, `token_sequence`, and `conflict_count`.
2. On `UNSAT` with `unsat_core: &[ClauseId]`:
   - Compute the Culprit Score for each `c` in `unsat_core`:
     `score(c) = c.mean_shannon_entropy * (-lambda * (current_clause - c.clause_id) as f64).exp() + gamma * (c.conflict_count as f64)`
   - Select culprit `c_star = argmax(score)`, pop solver and CST state to `c_star.clause_id - 1`, and record `c_star.token_sequence` in the Tabu prefix trie at `c_star.start_token_idx`.
3. Implement **Conflict-Driven Half-Space Domain Cut**:
   - Inspect the Farkas multiplier / interval directional derivative at `c_star.var_id`. If `c_star.committed_val` caused an upper-bound overflow downstream, tighten the active domain at depth `c_star.clause_id - 1` to `[lo, c_star.committed_val - tol]`; if it caused a lower-bound deficit, tighten to `[c_star.committed_val + tol, hi]`.
4. Emit a `ConflictBranchRecord` containing `(prefix_tokens, rejected_clause_tokens, unsat_core_ids, violation_margin)` so `veritas.rl` can export step-level DPO / Process-GRPO training pairs.
```

### Prompt 5: Build the 0-DoF Jump-Forward BPE Token-Healed Splicer & CRANE Controller
```text
You are an LLM inference systems engineer building `crates/veritas-core/src/zero_dof.rs`, `token_healer.rs`, and `crane_mode.rs`.
1. In `crane_mode.rs`:
   - Implement a lightweight state machine supporting `ReasoningMode::CraneCot { think_open: "<think>", think_close: "</think>" }`.
   - While in `<think>` mode, bypass Tier-0/1/2 masking so the LLM can perform unconstrained Chain-of-Thought, while passively scanning lines for `var = number` candidates to pre-warm hints. Immediately activate Tier-0/1/2 upon matching `</think>`.
2. In `zero_dof.rs` and `token_healer.rs`:
   - When the final input dependency of one or more topological `ZeroDofRule`s is bound at a JSON clause boundary, evaluate all cascaded 0-DoF expressions in topological order.
   - Format the deterministic JSON continuation string (e.g. `", \"expected_return\": 0.16500, \"volatility\": 0.19420, \"sharpe_ratio\": 0.64367"`).
   - Perform **BPE Boundary Token Healing**: decode the last 1-2 tokens of `token_ids_prefix` into bytes, concatenate with the deterministic splice string, re-tokenize using the model's fast tokenizer, and return `JumpForwardPrefill { backup_tokens, new_token_ids }` so `vLLM`/`SGLang`/`Transformers` can prefill the entire deterministic span in a single forward pass!
```

### Prompt 6: Build the Ecosystem Bridges, `veritas.sql` / `veritas.sec` Cartridges & Wasm Inspector
```text
You are a full-stack AI infrastructure engineer building `python/veritas/bridges/`, `python/veritas/cartridges/`, `python/veritas/rl/`, and `crates/veritas-wasm/`.
1. Implement `python/veritas/bridges/instructor_bridge.py` and `vllm_bridge.py`:
   - Provide `vt.patch_openai(client)` and `vt.INSTRUCTOR_MODE` so existing `Instructor` and OpenAI SDK users can pass `@vt.verified_schema` functions directly to `response_format` with zero code changes.
2. Implement `python/veritas/cartridges/sql.py` and `sec.py`:
   - `sql.py`: Accept a SQLAlchemy / DuckDB / SQLite schema metadata object and enforce valid table/alias/column scoping, foreign-key join compatibility, and mandatory tenant isolation predicates (`WHERE tenant_id = ...`) during SQL decoding.
   - `sec.py`: Implement `vt.WorldState` with finite-trace temporal logic (`vt.before(tool_a, tool_b)`, `vt.always(budget_spent <= max_budget)`) across multi-turn agent tool calls.
3. Implement `python/veritas/rl/collector.py`:
   - Convert recorded `ConflictBranchRecord`s and verified trajectories into HuggingFace `datasets` formatted for TRL `DPOTrainer` (step-level prefix preference pairs) and `GRPOTrainer` (dense process rewards).
4. Implement `python/veritas/cli/main.py` (`veritas check-spec`, `veritas verify`, and `veritas inspect trace.json` serving a single-page HTML/Wasm trace flamegraph & token entropy heatmap).
```
