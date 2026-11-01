# `veritas` — Neuro-Symbolic Solver-Guided Decoding & Formal Verification Engine
## Volume I: First-Principles Critique, Mathematical Formulation & Systems Architecture (`v2.0.0`)

**Document Status:** Principal Architecture & Mathematical Specification (`v2.0.0` — Upgraded from `v1.0.0-alpha`)  
**Target Stack:** Pure-Rust Native Solver Core (`veritas-core`: Embedded Simplex `QF_LRA`, Difference Logic `QF_IDL`, Affine Arithmetic `AF2` + `HC4-Revise`) + Incremental C-API Z3/cvc5 Bridge (`z3-sys`), Byte-Level Pushdown Automaton ([XGrammar](https://arxiv.org/pdf/2411.15100) / [llguidance](https://github.com/guidance-ai/llguidance)), Incremental GLR CST ([Tree-sitter](https://github.com/tree-sitter/tree-sitter)), Python 3.11+ ([Pydantic](https://www.google.com/search?q=https://docs.pydantic.dev/) v2, [SymPy](https://www.sympy.org/)), Zero-Copy DLPack Logit & KV-Cache Hooks ([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/), [SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/), `llama.cpp`, HuggingFace Transformers, and Masked Diffusion LLMs).

## 1. Executive Architecture Audit (`v1.0-alpha` $\rightarrow$ `v2.0` Upgrades)

A rigorous systems and theoretical audit of `veritas` `v1.0.0-alpha` against the 2025–2026 constrained decoding literature reveals seven foundational upgrades required to make `veritas` both theoretically watertight and orders of magnitude faster in production:

| Architectural Dimension | `v1.0.0-alpha` Limitation (Audited Flaw) | `v2.0.0` Breakthrough Mechanism | Primary Research & Systems Grounding |
| :--- | :--- | :--- | :--- |
| **1. Reasoning Expressivity vs. Format Lock-In** | Enforcing strict JSON from token 0 collapses Transformer circuit depth ($TC^0$) and degrades math/logic accuracy by 10–25%. | **CRANE Dual-Mode Trigger Switching (`<think>` $\rightarrow$ `<emit>`)**: Free CoT reasoning with passive equation mining, followed by strict solver-guided emission. | [CRANE (Banerjee et al., ICML 2025)](https://arxiv.org/abs/2502.09061), [Draft-Conditioned Decoding (2026)](https://arxiv.org/abs/2603.03305), [Nguyen et al. (2026)](https://arxiv.org/html/2610.35328v1) |
| **2. Local vs. Global Distribution Drift** | Stepwise greedy renormalization ($P_\theta(v) \mathbb{I}[v \in \mathcal{A}] / Z_t$) biases generation toward short-sighted dead-ends (Locally Constrained Decoding drift). | **Future-Volume Twisted SMC (`SHIM`-style $\gamma_t$ Twist)**: Weights candidate tokens by contracted interval box volume $\log \text{Vol}(D_{>t}(v))$ + $K$-particle clause resampling. | [Lightweight Bias Correction in Constrained Decoding (Ozgu et al., 2026)](https://starai.cs.ucla.edu/papers/OzguArxiv26.pdf), [Alignment in Constrained Code Gen (2026)](https://arxiv.org/html/2606.21619v1) |
| **3. 0-DoF BPE Tokenization & Field Ordering** | Out-of-order JSON keys prevent 0-DoF collapse; naive string injection causes BPE boundary token corruption (`": 0.17"` vs `":"`, `" 0.17"`). | **Topological Schema Key Reordering + Jump-Forward Token-Healed Prefill**: Sorts schema keys by dependency DAG and splices singular syntactic paths + 0-DoF values in 1 prefill burst. | [SGLang Jump-Forward Compressed FSM](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/), [llguidance Fast-Forward Tokens](https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md) |
| **4. Interval Dependency Blowup** | Standard `HC4-Revise` treats repeated variables independently ($x^2 - x \in [-1, 1]$ on $[0,1]$), yielding loose bounds on quadratic forms $w^T \Sigma w$. | **Hybrid Affine Arithmetic (`AF2`) + Cholesky Ellipsoidal Contractor**: Tracks first-order noise symbols $\epsilon_i$ across shared variables in $<35\ \mu\text{s}$. | IEEE-754 Outward-Rounded Interval & Affine Arithmetic (`inari` / `AF2`) |
| **5. Dynamic Collections & Open Containers** | Static base frame $F_0$ cannot pre-declare variables when `dict[str, float]` keys or `list[T]` lengths are dynamically chosen at runtime. | **Attribute Grammar Derivatives + Residual Accumulator Variables ($R_{\text{sum}}^{(k)}$)**: Incrementally synthesizes quantifier residuals over open AST containers. | [AGDecode: Semantic Attribute Grammar Derivatives (Maghera et al., 2026)](https://arxiv.org/html/2610.36105v1), [ChopChop (Nagy et al., 2025)](https://arxiv.org/abs/2509.00360) |
| **6. Impossible Specs & Over-Constrained Prompts** | If user inputs make the spec globally `UNSAT`, `v1.0` wastes `max_backtracks` LLM retries before crashing with an opaque error. | **0-Token Pre-Flight SAT Gate + Weighted MaxSMT (`@vt.prefer`)**: Proves global feasibility in $<2\text{ ms}$ *before* LLM prefill, returning minimal relaxation certificates on conflict. | [Microsoft Z3 Optimizer / MaxSMT](https://github.com/z3prover/z3) |
| **7. Installation & Cold-Start Friction** | Compiling `z3-sys` C++ from source takes 20+ minutes and adds 40MB+ binary bloat for simple linear/scheduling schemas. | **Tiered Native Rust Solver (`veritas-nano`) + Optional Z3 Plugin**: Ships a 4MB zero-dependency `abi3` wheel with pure-Rust Simplex (`QF_LRA`) + `QF_IDL` + `AF2` covering 90% of schemas natively. | [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) |

## 2. First-Principles Devil's Advocate Audit: The Seven Fatal Traps & Structural Breakthroughs

Coupling token-level LLM logit masking with formal verification engines (SMT, CAS, and interval contractors) in real time will fail in production unless seven fundamental systems bottlenecks are solved from first principles.

### Trap 1: The Subword-to-Symbol Impedance Mismatch (Why Token-Level Z3 Calls Fail)
* **The Flaw:** An LLM emits BPE subword tokens (`"0"`, `"."`, `"3"`, `"5"`, `","`), not atomic mathematical symbols or complete AST nodes. Invoking [Microsoft Research's Z3 Theorem Prover](https://github.com/z3prover/z3) on every token step over a 128,256-token vocabulary ($|\mathcal{V}| = 128,256$) at $0.5\text{ ms}$ per check would require $64\text{ seconds}$ per token—destroying GPU inference throughput.
* **The `veritas` Breakthrough — Three-Tier Lazy Clause-Boundary Gating + Prefix Affine Contractors:**
  1. **Tier-0 Syntactic Pushdown Bitmasking ($<12\ \mu\text{s}$):** Uses a pre-compiled byte-level Pushdown Automaton (building on [XGrammar](https://arxiv.org/pdf/2411.15100) context-independent token partitioning and [llguidance](https://github.com/guidance-ai/llguidance) lexeme tries) to enforce context-free syntax (JSON schema, Python AST, SMILES grammar) at every token via SIMD bitwise AND.
  2. **Tier-1 Ultra-Fast Numeric Prefix Interval & Affine Propagation ($<35\ \mu\text{s}$):** While the LLM is mid-number (e.g., emitting digits of `p.weights["AAPL"] = 0.3...`), `veritas` never invokes Z3. Instead, a zero-allocation Rust **Hybrid HC4-Revise + Affine Arithmetic (`AF2`) Contractor** maintains tight lower and upper bounds $[L_i, U_i]$ on every partially generated numeric literal and masks out infeasible leading digits in $O(1)$ time (e.g., if $\sum_{k=1}^{i-1} w_k = 0.85$ and $w_i$ starts with `"0."`, digits `'2'` through `'9'` are masked immediately without touching Z3).
  3. **Tier-2 Semantic Clause-Boundary SMT Verification ($0.3\text{–}4.5\text{ ms}$):** Only when [Tree-sitter](https://github.com/tree-sitter/tree-sitter) detects a **completed semantic clause** (a closed key-value pair, a completed equation line, or an AST statement boundary) does `veritas` lower the completed AST delta into an incremental solver frame (`veritas-nano` pure-Rust Simplex/IDL solver for linear/difference logic in $<150\ \mu\text{s}$, or incremental `z3::Solver::push()` for non-linear/quantified theories in $0.5\text{–}4.5\text{ ms}$).

### Trap 2: The Non-Linear Real Arithmetic (`QF_NRA`), Dependency Blowup & Floating-Point Trap
* **The Flaw:** Consider the portfolio Sharpe ratio constraint:
  $$\text{sharpe\_ratio} == \frac{\text{expected\_return} - 0.04}{\sqrt{w^T \Sigma w}}$$
  This contains a square root of a quadratic form ($w^T \Sigma w$). In SMT-LIB, exact IEEE-754 floating-point theory (`QF_FPA`) bit-blasts floats into 64-bit circuits and hangs on square roots, while Cylindrical Algebraic Decomposition (CAD) in `QF_NRA` has doubly-exponential worst-case complexity $O(2^{2^n})$ and cannot natively evaluate transcendental functions ($\exp, \ln, \sin$). Furthermore, naive interval arithmetic suffers from the **Interval Dependency Problem**: evaluating $w_1^2 - 0.5 w_1$ by treating the two occurrences of $w_1 \in [0.05, 0.50]$ independently produces overly wide bounds.
* **The `veritas` Breakthrough — Affine Arithmetic (`AF2`) + Polynomial Lifting + 0-DoF Jump-Forward Collapse:**
  1. **Affine Arithmetic (`AF2`) + Cholesky Quadratic Envelopes:** Every constrained real variable $x_i \in [l_i, u_i]$ is represented in affine form $\hat{x}_i = x_{i,0} + \sum_{k=1}^n x_{i,k} \epsilon_k + x_{i,\pm} \epsilon_{\pm}$ with noise symbols $\epsilon_k \in [-1, 1]$. Linear combinations cancel shared noise symbols exactly, while positive semi-definite quadratic forms $w^T \Sigma w = \| L^T w \|_2^2$ are contracted via precomputed Cholesky factor $L$ in $O(n^2)$ operations without dependency blowup.
  2. **Polynomial Squaring & SymPy Canonicalization:** At decoration time (`@vt.verified_schema`), `veritas` parses the Python AST via a [SymPy](https://www.sympy.org/) canonicalizer, automatically algebraicizing fractional and root expressions where sign invariants hold (e.g., rewriting $S = (R - r_f)/\sqrt{w^T \Sigma w}$ into $S^2 \cdot (w^T \Sigma w) - (R - r_f)^2 \in [-\epsilon, +\epsilon]$ along with $\text{sign}(S) = \text{sign}(R - r_f)$) and delegating transcendental functions to a Rust **Interval Newton Contractor**.
  3. **Topological Key Ordering & 0-DoF Jump-Forward Token Injection:** Once `weights` is generated, `expected_return`, `volatility`, and `sharpe_ratio` have **zero degrees of freedom (0-DoF)**. `veritas` topologically sorts the JSON schema keys at compile time so free variables (`weights`) are always emitted first. The instant `weights` closes, `veritas` evaluates all downstream 0-DoF expressions in exact IEEE-754 / arbitrary-precision rational arithmetic, splices the entire JSON substring `", \"expected_return\": 0.165, \"volatility\": 0.1942, \"sharpe_ratio\": 0.6437}"` through a **BPE Token-Healing Jump-Forward Prefill** ([SGLang](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)), and validates any post-conditions (`sharpe_ratio >= 0.65`) in $<5\ \mu\text{s}$ before the GPU even runs a forward pass!

### Trap 3: Chronological Backtracking Loops vs. Conflict-Driven Non-Chronological Jumpback
* **The Flaw:** If an LLM generates 8 asset weights $w_1, \dots, w_8$ each around $0.20$, a naive chronological backtrack of $K=16$ tokens only erases $w_8$ and $w_7$, leaving $\sum_{i=1}^6 w_i = 1.20 > 1.0$, which is still `UNSAT`—causing an infinite loop at $w_7$.
* **The `veritas` Breakthrough — UnsatCore Token Attribution & Entropy-Weighted CDCL Jumpback:**
  Every clause assertion pushed at AST node $u_k$ (spanning token indices $[s_k, e_k]$) is tracked with a named Boolean assumption literal $\alpha_k$. When `solver.check()` or `interval.propagate()` returns `UNSAT`, `veritas` extracts the minimal **UnsatCore** $\mathcal{C} = \{\alpha_{k_1}, \dots, \alpha_{k_m}\}$, identifies the **Entropy-Weighted Conflict Culprit** $k^* \in \mathcal{C}$, rewinds the [vLLM PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention/) or [SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) KV-cache directly to $s_{k^*} - 1$, and attaches both a **Tabu Prefix Mask** (banning the exact failed token sequence) and **Domain-Cut Learning** (tightening the interval bound at $u_{k^*}$ so the entire infeasible half-space is pruned on retry!).

### Trap 4: The "Format Restriction Reasoning Penalty" (`CRANE` & Circuit Complexity Trap)
* **The Flaw:** As demonstrated empirically by [Tam et al. (2024)](https://www.researchgate.net/publication/382885370_Let_Me_Speak_Freely_A_Study_on_the_Impact_of_Format_Restrictions_on_Performance_of_Large_Language_Models) and proven theoretically by [CRANE (Banerjee et al., ICML 2025)](https://arxiv.org/abs/2502.09061) and [Draft-Conditioned Constrained Decoding (2026)](https://arxiv.org/abs/2603.03305), forcing a constant-depth Transformer to emit constrained structured tokens immediately from step 0 strips away its intermediate scratchpad computation, reducing accuracy on multi-step symbolic and arithmetic tasks by up to 20%.
* **The `veritas` Breakthrough — CRANE Dual-Mode Trigger Switching & Passive Scratchpad Mining:**
  `veritas` v2.0 natively supports **Reasoning-Augmented Constrained Decoding**:
  1. **Unconstrained Scratchpad Window (`<think>...</think>` or inline `/* ... */`):** The model reasons freely during its Chain-of-Thought phase (or switches dynamically into unconstrained scratchpad windows before complex sub-clauses as in [CRANE](https://icml.cc/virtual/2025/poster/43624) and [Nguyen et al., 2026](https://arxiv.org/html/2610.35328v1)).
  2. **Passive Scratchpad Equation Mining:** While the LLM reasons inside `<think>`, `veritas`'s ShadowParser passively extracts candidate numerical derivations and checks dimensional/algebraic consistency in the background, pre-warming the `ZeroDof` cache.
  3. **Zero-Latency Trigger Switch:** The instant the model emits `</think>` (or the grammar start delimiter), `veritas` engages Tier-0/1/2 enforcement and 0-DoF Collapse—combining the full reasoning expressivity of unconstrained CoT with $100\%$ formal verification guarantees on the structured output.

### Trap 5: Locally Constrained Decoding (LCD) Myopia & Distribution Distortion
* **The Flaw:** Masking invalid tokens locally at step $t$ and dividing by $Z_t(y_{\le t}) = \sum_{v \in \mathcal{A}(y_{\le t})} P_\theta(v \mid y_{\le t})$ samples from the **Locally Constrained Distribution** $q_{\text{LCD}}(y)$, NOT the true global posterior $P_\theta(y \mid y \in \mathcal{L}_{\mathcal{V}}(\mathcal{G}, \Phi))$. As proven in [Lightweight Bias Correction in Constrained Decoding (Ozgu et al., Aug 2026)](https://starai.cs.ucla.edu/papers/OzguArxiv26.pdf) and [The Alignment Problem in Constrained Code Generation (June 2026)](https://arxiv.org/html/2606.21619v1), local masking ignores whether a valid token $v$ leads to a tiny, nearly-empty future feasible region versus a wide, high-probability feasible region.
* **The `veritas` Breakthrough — Volume-Twisted SMC (`SHIM`-Style Future-Mass Correction):**
  `veritas` corrects local masking bias at zero training cost by multiplying each admissible token's probability by an analytical **Future Feasibility Twist Factor** $\gamma_t(v \mid y_{\le t})$ derived from the contracted hyper-volume of remaining free variables after appending $v$:
  $$\tilde{P}_{\text{Veritas}}(v \mid y_{\le t}) \propto P_\theta(v \mid y_{\le t}) \cdot \mathbb{I}\!\left[v \in \mathcal{A}(y_{\le t})\right] \cdot \left(\frac{\text{Vol}\!\left(\text{HC4}(D_t \mid y_{\le t} \cdot v)\right)}{\text{Vol}(D_t)}\right)^{\beta_{\text{twist}}}$$
  Tokens that pinch remaining variables into a razor-thin or empty corner of the polytope receive a smooth, mathematically principled probability discount *before* a hard `UNSAT` collision occurs!

### Trap 6: Dynamic Collections (`dict`, `list`) vs. Static Symbolic Variables
* **The Flaw:** Real-world Pydantic schemas contain variable-length lists (`list[Order]`) and dynamic dictionaries (`dict[str, float]`). If base frame $F_0$ only declares static fields, Existential Lookahead fails on dynamic containers.
* **The `veritas` Breakthrough — Attribute Grammar Derivatives & Residual Accumulators ([AGDecode, Oct 2026](https://arxiv.org/html/2610.36105v1)):**
  For any quantifier or reduction over a collection $C$ with cardinality bounds $N \in [N_{\min}, N_{\max}]$ and element bounds $x \in [l, u]$ (such as $\sum_{x \in C} x = S$ and $\forall x \in C, l \le x \le u$), `veritas` maintains a **Symbolic Residual Accumulator** $R_k = S - \sum_{i=1}^k x_i$ after $k$ elements have been emitted. Existential lookahead over the ungenerated suffix reduces to the closed-form disjunctive interval check:
  $$\exists m \in [\max(0, N_{\min} - k),\; N_{\max} - k] \cap \mathbb{Z} \;\;\text{s.t.}\;\; R_k \in [m \cdot l,\; m \cdot u]$$
  This executes in $O(1)$ CPU instructions ($<2\ \mu\text{s}$) during Tier-1 prefix propagation and is asserted incrementally in Tier-2!

### Trap 7: Cloud-Hosted Black-Box APIs vs. Local Open-Weight Models
* **The Flaw:** Closed APIs (OpenAI, Anthropic, Gemini API) do not expose per-token mutable logit callbacks or KV-cache block-table pointers.
* **The `veritas` Breakthrough — Dual Execution Modes + Speculative Prefix-Restart:**
  * **Mode A: Native White-Box Logit Engine (`veritas.LogitsProcessor` + KV Manager):** Full zero-copy DLPack token masking, Jump-Forward 0-DoF prefill, and Paged/Radix KV-cache rewind for open-weight models (`vLLM`, `SGLang`, `llama.cpp`, `HuggingFace`, `MLX`, and diffusion models like LLaDA/Dream).
  * **Mode B: Speculative Clause-Stream Supervisor (`veritas.AsyncStreamSupervisor`):** For black-box SSE APIs (or free-tier Groq / HuggingFace serverless endpoints), `veritas` parses the incoming SSE token stream through the incremental Tree-sitter + Tier-1/Tier-2 engine, aborts the HTTP stream within $<1\text{ ms}$ of an `UNSAT` clause, injects the exact `UnsatCore` interval bound and computed `0-DoF` values into the assistant prefix, and resumes generation using provider Prefix Caching.

## 3. Formal Mathematical Framework

### 3.1 Neuro-Symbolic Constrained Language Intersection

Let $\Sigma$ be the byte alphabet and $\mathcal{V} \subset \Sigma^+$ be the subword token vocabulary of an autoregressive or diffusion language model $M_\theta$. Let $L_{\text{CFG}}(\mathcal{G}) \subseteq \Sigma^*$ be the context-free language induced by the structural grammar $\mathcal{G}$ (potentially augmented with unconstrained reasoning productions $R_{\text{think}}$ as in [CRANE](https://arxiv.org/abs/2502.09061)), and let $\Phi(z) = \Phi_{\text{hard}}(z) \wedge \Psi_{\text{soft}}(z)$ be a first-order formula over background theories $\mathcal{T} = \mathcal{T}_{\text{LRA}} \cup \mathcal{T}_{\text{NRA}} \cup \mathcal{T}_{\text{IDL}} \cup \mathcal{T}_{\text{BV}} \cup \mathcal{T}_{\text{FP}} \cup \mathcal{T}_{\text{UF}}$ evaluated on the semantic interpretation $z = \llbracket y \rrbracket_{\text{AST}}$ of string $y \in \Sigma^*$.

The target **Veritas Language** $\mathcal{L}_{\mathcal{V}}(\mathcal{G}, \Phi_{\text{hard}})$ is the formal intersection:
$$\mathcal{L}_{\mathcal{V}}(\mathcal{G}, \Phi_{\text{hard}}) = \left\{ y = (v_1, v_2, \dots, v_T) \in \mathcal{V}^* \;\middle|\; \text{concat}(y) \in L_{\text{CFG}}(\mathcal{G}) \;\wedge\; \mathcal{T} \models \Phi_{\text{hard}}\!\left(\llbracket \text{concat}(y) \rrbracket_{\text{AST}}\right) \right\}$$

Using the **Attribute Grammar Derivative** operator $\partial_{v} \Phi$ ([AGDecode, Maghera et al., 2026](https://arxiv.org/html/2610.36105v1)), each partial token prefix $y_{\le t}$ induces a residual semantic specification $\Phi^{(t)} = \partial_{y_{\le t}} \Phi_{\text{hard}}$ over the partially constructed syntax tree $T(y_{\le t})$ with existentially quantified hole variables $\mathbf{x}_{\text{free}}$. A candidate token $v \in \mathcal{V}$ is **realizable** at step $t+1$ if and only if:
$$\mathcal{A}(y_{\le t}) = \left\{ v \in \mathcal{V} \;\middle|\; \text{concat}(y_{\le t} \cdot v) \in \text{Prefix}\!\left(L_{\text{CFG}}(\mathcal{G})\right) \;\wedge\; \mathcal{T} \models \exists \mathbf{x}_{\text{free}} \in D_t,\; \partial_{v} \Phi^{(t)}(\mathbf{x}_{\text{free}}) \right\}$$

`veritas` factors the indicator and twist mask $m_t \in [-\infty, 0]^{|\mathcal{V}|}$ into four hierarchical operators:

$$m_t(v) = m_t^{\text{PDA}}(v) + m_t^{\text{AF2-Box}}(v) + m_t^{\text{SMT}}(v) + \beta_{\text{twist}} \log \gamma_t^{\text{Vol}}(v)$$

1. **Syntactic PDA Mask $m_t^{\text{PDA}}(v) \in \{0, -\infty\}$:** Exact membership in $\text{Prefix}(L_{\text{CFG}}(\mathcal{G}))$, computed in $O(1)$ amortized time via context-independent vocabulary bitmasks and pushdown stack caches ([XGrammar](https://arxiv.org/pdf/2411.15100)).
2. **Affine-Box Contractor Mask $m_t^{\text{AF2-Box}}(v) \in \{0, -\infty\}$:** Given the current contracted domain $D_t = [l_1, u_1] \times \dots \times [l_d, u_d]$ computed by the **HC4-Revise + Affine (`AF2`)** operator on the active constraint DAG, any token $v$ extending a numeric literal $x_i$ defines a prefix completion interval $[\underline{\text{val}}(x_i \cdot v), \overline{\text{val}}(x_i \cdot v)]$. If $[\underline{\text{val}}(x_i \cdot v), \overline{\text{val}}(x_i \cdot v)] \cap [l_i, u_i] = \emptyset$, then $m_t^{\text{AF2-Box}}(v) = -\infty$.
3. **Future-Volume Bias Correction Twist $\gamma_t^{\text{Vol}}(v) \in (0, 1]$:** For numeric continuation tokens $v$ where $m_t^{\text{AF2-Box}}(v) = 0$, estimates the residual polytope volume ratio after propagating $x_i \in [\underline{\text{val}}(x_i \cdot v), \overline{\text{val}}(x_i \cdot v)] \cap [l_i, u_i]$ through one step of linear `HC4-Revise`, counteracting Locally Constrained Decoding myopia ([Ozgu et al., 2026](https://starai.cs.ucla.edu/papers/OzguArxiv26.pdf)).
4. **Clause-Boundary SMT Assertion $m_t^{\text{SMT}}(v) \in \{0, -\infty\}$:** Triggered only when appending $v$ closes a semantic AST node $u_k$. Evaluates $\text{SAT}\!\left(\Phi_{\le k-1} \wedge \phi(u_k) \wedge \exists \mathbf{x}_{\text{free}} \in D_t,\; \Phi_{\text{lookahead}}(\mathbf{x}_{\text{free}})\right)$.

### 3.2 Pre-Flight Static Feasibility Gate & Minimum Relaxation Certificates

Before running LLM token generation, `veritas` executes a **0-Token Pre-Flight Verification Pass** on the base frame $F_0$:
1. Run `HC4-Revise + AF2` to fixed point on $D_0$, followed by `solver.check()` on $F_0$.
2. If $F_0$ is `UNSAT` (meaning the user's schema constraints and input constants are mutually contradictory before a single token is generated), `veritas` immediately aborts LLM invocation and runs a **Slack-Variable L1 Relaxation LP/MaxSMT**:
   $$\min_{\mathbf{x}, \boldsymbol{\xi} \ge 0} \sum_{m=1}^M w_m \xi_m \quad \text{s.t.} \quad g_m(\mathbf{x}) \le \xi_m \quad \forall m \in \{1, \dots, M\}$$
3. Returns an immediate `InfeasibleSpecError` containing the **Minimal UnsatCore** of conflicting user rules and the exact **Pareto Slack Witness** $\boldsymbol{\xi}^*$ required to make the specification satisfiable.

### 3.3 Bipartite 0-DoF Dependency Graph & Jump-Forward BPE Splicing

Let $\mathcal{G}_{\text{dep}} = (\mathcal{X} \cup \mathcal{E}, E)$ be the bipartite graph between schema variables $\mathcal{X} = \{x_1, \dots, x_n\}$ and equality/derivation constraints $\mathcal{E} = \{e_1, \dots, e_r\}$.
* **Compile-Time Topological Ordering:** If $e_m \in \mathcal{E}$ defines $x_j := f_m(x_{i_1}, \dots, x_{i_k})$ (via `vt.derive(...)` or functional equality detection in SymPy), `veritas` directs edges $x_{i_p} \to e_m \to x_j$. The JSON grammar generator orders schema keys according to a topological sort of $\mathcal{G}_{\text{dep}}$ so that $\forall p,\; \text{pos}(x_{i_p}) < \text{pos}(x_j)$.
* **Runtime Jump-Forward BPE Splice:** When the last upstream input $x_{i_k}$ closes at byte offset $b$, `veritas`:
  1. Evaluates $c_j = f_m(c_{i_1}, \dots, c_{i_k})$ in IEEE-754 / rational arithmetic, rounded to `precision_digits` so that $|c_j - f_m(\cdot)| \le \text{tol}$.
  2. Concatenates the deterministic syntactic transition string (e.g., `", \"expected_return\": "`) with the formatted literal $\text{str}(c_j)$ (and any subsequent cascaded 0-DoF fields!).
  3. Applies **BPE Boundary Token Healing**: backs up 1 token if the last emitted token was a partial BPE prefix of the splice string, re-tokenizes the combined byte span using the model's exact tokenizer, and executes a single batched KV-cache extend pass.

### 3.4 Entropy-Weighted CDCL Jumpback & Domain-Cut Learning

When clause boundary $k$ yields `UNSAT` with minimal `UnsatCore` $\mathcal{C} \subseteq \{\alpha_1, \dots, \alpha_k\}$:
1. **Culprit Score Calculation:** For each AST assignment node $u_j$ with $\alpha_j \in \mathcal{C}$ spanning tokens $t \in [s_j, e_j]$ with recorded step entropies $H_t = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)$:
   $$S(u_j) = \left(\frac{1}{e_j - s_j + 1} \sum_{t=s_j}^{e_j} H_t\right) \cdot \exp\!\left(-\lambda (k - j)\right) + \gamma \cdot \text{ConflictCount}(u_j)$$
2. **Non-Chronological Jumpback ($j^* = \arg\max_{\alpha_j \in \mathcal{C}} S(u_j)$):**
   * Pop solver frames from $F_k$ down to $F_{j^*-1}$.
   * Rewind the incremental Tree-sitter CST to $\text{byte\_start}(u_{j^*})$ and truncate the PagedAttention/RadixAttention KV-cache block table to $s_{j^*} - 1$.
3. **Conflict-Driven Domain Cut (Half-Space Pruning):**
   * Beyond adding the exact token sequence $(v_{s_{j^*}}, \dots, v_{e_{j^*}})$ to the Tabu trie at $u_{j^*}$, if the violated constraint $g(x_{j^*}, \mathbf{x}_{-j^*}) \le 0$ is monotonic in $x_{j^*}$ over $D_{j^*}$ (e.g., $x_{j^*} = c_{j^*}$ was too large to leave room for downstream variables), `veritas` permanently shrinks the interval bound at depth $j^*-1$ to $U_{j^*}^{\text{new}} = c_{j^*} - \delta_{\text{cut}}$ (or $L_{j^*}^{\text{new}} = c_{j^*} + \delta_{\text{cut}}$). Thus, the LLM cannot even sample a nearby value like $c_{j^*} - 10^{-5}$ that would fail for the exact same algebraic reason!

### 3.5 Non-Sequential Masked Diffusion LLM Intersection (DINGO + CFG + Factor-Graph Contractors)

Building on [DINGO (NeurIPS 2025)](https://openreview.net/forum?id=KaYMGsnZ4R) and [Constrained Decoding of Diffusion LLMs with Context-Free Grammars (Mündler, Dekoninck, & Vechev, 2025/2026)](https://openreview.net/forum?id=7Sph4KyeYO), masked diffusion language models (LLaDA, Dream, Mercury) iteratively unmask positions in a partially masked sequence $W^{(\tau)} = w_1 \; [\text{MASK}]^{k_1} \; w_2 \dots w_m$.

`veritas` generalizes syntactic CYK/Earley span charts to **Neuro-Symbolic Factor-Graph Unmasking**:
1. **Unbiased Gumbel-Max DP Sampling over Span Charts:** Addressing the mode-collapse critique of deterministic Viterbi decoding in [DINGO](https://openreview.net/forum?id=KaYMGsnZ4R), `veritas` perturbs diffusion log-potentials with Gumbel noise $G_{i,v} \sim \text{Gumbel}(0, T_{\text{temp}})$ before running inside-outside dynamic programming over the DFA/CFG chart, guaranteeing **unbiased temperature sampling** conditioned on the formal language.
2. **Bidirectional Interval Propagation Across Masked Holes:** Because diffusion models can unmask a downstream field (e.g., `volatility = 0.18`) *before* upstream `weights` are unmasked, `veritas`'s `HC4-Revise` contractor propagates bounds **bidirectionally** across the factor graph at every diffusion step $\tau$.
3. **Surgical Culprit Re-Masking:** If unmasking island $w_r$ triggers `UNSAT` with previously unmasked islands, `veritas` queries the `UnsatCore` $\mathcal{C}$ and **re-masks only the lowest-confidence island** $u_{j^*} \in \mathcal{C}$ back to $[\text{MASK}]^{\text{len}(u_{j^*})}$, preserving all valid unmasked islands across the sequence.

## 4. End-to-End System Architecture (`veritas` v2.0 Engine)

### 4.1 Four-Layer Component Topology

1. **Layer 0: Pre-Flight Compiler & Symbolic Analyzer (`veritas.compiler`)**
   * Introspects `@vt.verified_schema` functions via Python `ast` + Pydantic v2 schemas.
   * Performs **Topological Key Reordering** so 0-DoF derived fields always follow their inputs.
   * Runs the **0-Token Pre-Flight Feasibility Gate** and compiles constraints into portable **Veritas Intermediate Representation (`VIR`)** bytecode.

2. **Layer 1: Pure-Rust Native Fast-Path Engine (`veritas-core::native`)**
   * **`AutomatonEngine`:** Precomputed context-independent token bitmasks + byte-level PDA stack ([XGrammar](https://arxiv.org/pdf/2411.15100) / [llguidance](https://github.com/guidance-ai/llguidance)).
   * **`ShadowParser`:** Incremental `tree-sitter` C parsers (JSON, Python, Rust, SQL, SMILES, LaTeX) + `<think>`/`<emit>` mode-switch detector.
   * **`AffineIntervalContractor`:** Zero-allocation IEEE-754 outward-rounded `HC4-Revise` + `AF2` noise-symbol engine + Cholesky quadratic contractor.
   * **`VeritasNanoSolver`:** Embedded pure-Rust incremental Dual Simplex (`QF_LRA`) + Bellman-Ford Difference Logic (`QF_IDL`) solver with native `UnsatCore` tracking—handling $90\%$ of production schemas in $<150\ \mu\text{s}$ with **zero external C++ dependencies**.

3. **Layer 2: Heavyweight SMT & CAS Oracle Pool (`veritas-core::oracles`)**
   * **`IncrementalSmtPool`:** Thread-safe pool of `z3::Solver` and `cvc5` instances for quantified formulas, bitvectors (`QF_BV`), algebraic numbers (`QF_NRA`), and Weighted MaxSMT (`vt.prefer`), plus [Rocq/Lean4](https://arxiv.org/html/2610.36105v1) and SMT-LIB2 proof certificate emitters.

4. **Layer 3: Zero-Copy Inference & RL Bridge (`veritas.bridges` & `veritas.rl`)**
   * Universal `VeritasLogitsProcessor` + `KVCheckpointManager` for `vLLM`, `SGLang`, `llama.cpp`, `Transformers`, `MLX`, and Diffusion LLMs via zero-copy `DLPack` tensors.
   * **`VeritasRLCollector`:** Automatically exports every CDCL jumpback (`UnsatCore` rejected prefix vs. verified completion) as step-level **DPO / GRPO preference tuples** for post-training reasoning models.

## 5. Core Algorithmic Specifications (Production Pseudocode)

### Algorithm 1: CRANE Mode-Switched Decoding, 0-DoF Jump-Forward & Volume-Twisted Masking

```python
def step_logits_mask(
    state: VeritasDecodeState,
    token_ids_prefix: list[int],
    logits: Tensor,              # In-place DLPack view, shape: [vocab_size]
) -> Tuple[Tensor, Optional[DecoderAction]]:
    # 0. CRANE Dual-Mode Check: If inside unconstrained <think>...</think> window,
    #    allow full vocabulary while passively mining candidate equations in background.
    if state.in_reasoning_scratchpad():
        state.passive_scratchpad_miner.observe_prefix(token_ids_prefix)
        return logits, None

    # 1. Zero-DoF Jump-Forward Collapse + BPE Token-Healing Splice
    active_slot = state.shadow_parser.current_semantic_slot()
    if active_slot is not None and state.dependency_graph.is_zero_dof_ready(active_slot):
        splice_tokens = state.zero_dof_engine.compute_token_healed_splice(
            active_slot, last_token_id=token_ids_prefix[-1]
        )
        # Return a JumpForwardPrefill action that extends the KV-cache in 1 batch pass
        return logits, JumpForwardPrefill(token_ids=splice_tokens)

    # 2. Tier-0: Syntactic PDA Bitmask (< 12 us via AVX-512 / NEON)
    pda_bitmask = state.automaton.get_valid_token_bitmask(state.pda_state)
    apply_bitmask_simd(logits, pda_bitmask)

    # 3. Tier-1: Affine-Box (HC4 + AF2) Contractor + Future-Volume Twist (< 35 us)
    if active_slot is not None and active_slot.is_numeric_literal():
        prefix_str = active_slot.current_literal_prefix()
        bounds: Interval = state.contractor.get_bounds(active_slot.var_id)
        base_log_vol = state.contractor.current_log_volume()

        for token_id in state.numeric_continuation_tokens(pda_bitmask):
            cand_prefix = prefix_str + state.vocab.decode(token_id)
            cand_range = parse_numeric_prefix_interval(cand_prefix, bounds)
            if cand_range.is_empty():
                logits[token_id] = -inf
            elif state.config.volume_twist_beta > 0.0:
                # SHIM-style lightweight bias correction for future feasibility mass
                rem_log_vol = state.contractor.estimate_log_volume_after(
                    active_slot.var_id, cand_range
                )
                logits[token_id] += state.config.volume_twist_beta * (rem_log_vol - base_log_vol)

    # 4. Immediate Dead-End Detection -> Trigger Non-Chronological CDCL Jumpback
    if torch.all(logits == -inf):
        return logits, state.trigger_cdcl_jumpback(reason="EMPTY_CONTRACTED_DOMAIN")

    return logits, None
```

### Algorithm 2: Post-Token Clause Verification, Domain-Cut Learning & RL Trace Harvesting

```python
def on_token_sampled(
    state: VeritasDecodeState,
    sampled_token_id: int,
    token_entropy: float,
) -> Optional[DecoderAction]:
    token_bytes = state.vocab.decode_bytes(sampled_token_id)
    state.pda_state = state.automaton.step(state.pda_state, sampled_token_id)
    completed_clauses = state.shadow_parser.feed_bytes(
        token_bytes, token_idx=state.step_idx, entropy=token_entropy
    )

    if not completed_clauses:
        return None

    for clause in completed_clauses:
        # 1. Fast Tier-1 Affine + Residual Accumulator Propagation (< 35 us)
        if not state.contractor.propagate_clause(clause):
            core_nodes = state.contractor.extract_conflict_core(clause)
            state.rl_collector.record_conflict_branch(state, clause, core_nodes)
            return state.rewind_with_domain_cut(core_nodes, failed_clause=clause)

        # 2. Tier-2 Solver Check: Pure-Rust VeritasNano (< 150 us) or Incremental Z3 (< 4 ms)
        solver = state.select_tier2_oracle(clause)
        solver.push()
        tracker_lit = solver.fresh_bool_tracker(clause.node_id)
        solver.assert_and_track(state.vir_lowering.bind_clause(clause), tracker_lit)

        status = solver.check_with_timeout(timeout_ms=state.config.smt_timeout_ms)
        if status == SolverStatus.UNSAT:
            unsat_core = solver.get_unsat_core()
            core_nodes = [state.node_from_tracker(lit) for lit in unsat_core]
            state.rl_collector.record_conflict_branch(state, clause, core_nodes)
            return state.rewind_with_domain_cut(core_nodes, failed_clause=clause)

        state.commit_checkpoint(clause)

    return None
```

## 6. Performance, Latency & Memory Engineering Budget (`v2.0`)

| Operation | Frequency | Latency Target (`v2.0`) | Mechanism |
| :--- | :--- | :--- | :--- |
| **0-Token Pre-Flight Feasibility Gate** | Once per request ($F_0$) | $0.4\text{–}1.8\text{ ms}$ | Static `HC4 + AF2` + base-frame SAT/LP check before LLM prefill |
| **Tier-0 Syntactic PDA Mask** | Every constrained token | $6\text{–}12\ \mu\text{s}$ | Context-independent bitmask cache + AVX-512 / NEON bitwise AND |
| **Incremental Tree-sitter CST Edit** | Every token ($100\%$) | $5\text{–}10\ \mu\text{s}$ | Incremental GLR subtree reuse via `ts_tree_edit` |
| **Tier-1 Affine + Box Contractor + Twist** | Numeric tokens ($\sim 15\%$) | $18\text{–}35\ \mu\text{s}$ | Zero-allocation Rust `HC4-Revise` + `AF2` + Cholesky bound + $\gamma_t^{\text{Vol}}$ |
| **Tier-2A `VeritasNano` (Simplex / IDL)** | Linear clause boundaries | $40\text{–}150\ \mu\text{s}$ | Pure-Rust Incremental Dual Simplex + Bellman-Ford Difference Logic |
| **Tier-2B Incremental Z3 / cvc5 Check** | Non-linear / quantified clauses | $0.5\text{–}4.5\text{ ms}$ | Persistent `z3::Solver` pool with existential lookahead in $F_0$ |
| **0-DoF Jump-Forward Token-Healed Burst** | Derived variables ($\sim 15\text{–}40\%$) | $<5\ \mu\text{s}$ CPU (skips 10–30 LLM steps!) | Topological key ordering + exact Rust eval + single batched KV prefill |
| **Net End-to-End Throughput Impact** | Overall generation | **$1.15\times\text{–}1.40\times$ FASTER than unconstrained** | 0-DoF Jump-Forward skips more GPU forward passes than solver checks cost |
