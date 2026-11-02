# What's Next for Veritas

## Immediate Follow-ups (Post-PR)

### 1. Documentation Updates
- [ ] Update `docs/NATIVE.md` with quadratic_form wire format spec
- [ ] Update `README.md` with new features (GBNF, certificate signing, dynamic limits)
- [ ] Add usage examples for certificate signing in handbook

### 2. Native Engine Parity
- [ ] Add quadratic_form to PyO3 bindings (`crates/veritas-py/src/lib.rs`)
- [ ] Add quadratic_form to WASM bindings (`crates/veritas-wasm/src/lib.rs`)
- [ ] Extend native backend parity tests (`tests/test_native_backend.py`)

### 3. CI/CD
- [ ] Add `cryptography` to dev dependencies for certificate signing tests
- [ ] Verify GBNF generation works in CI environment
- [ ] Add property tests for quadratic form contraction

---

## Medium-term Roadmap

### 4. Quadratic Contraction Improvements
- [ ] Full interval-based quadratic solving (not just sampling)
- [ ] Support for quadratic inequalities (Le, Ge, Lt, Gt)
- [ ] Integration with affine arithmetic (AF2) for tighter bounds

### 5. Pattern Masking Enhancements
- [ ] Full word boundary support with context tracking
- [ ] Lookahead/lookbehind support (with clear fallback semantics)
- [ ] Unicode property escapes (\p{...}, \P{...})

### 6. GBNF Enhancements
- [ ] Numeric range constraints in GBNF (using regex character classes)
- [ ] String length constraints (minLength/maxLength)
- [ ] Pattern constraints (translate supported regex subset to GBNF)
- [ ] Enum value constraints

### 7. Certificate Enhancements
- [ ] Certificate chains / trust anchors
- [ ] Timestamping support
- [ ] Revocation mechanism
- [ ] Hardware key support (HSM, TPM, WebAuthn)

### 8. Performance Optimizations
- [ ] SIMD-accelerated interval arithmetic
- [ ] Parallel contractor propagation
- [ ] Incremental GBNF compilation caching
- [ ] Native contractor for full constraint set (not just mask/numeric_prefix)

---

## Long-term Vision

### 9. Formal Verification
- [ ] Coq/Lean proofs for core soundness theorems
- [ ] Verified Rust compiler pipeline (using `creusot` or `prusti`)
- [ ] SMT solver integration as oracle (Z3, cvc5)

### 10. Domain Cartridges Expansion
- [ ] ML model verification (ONNX, TensorFlow, PyTorch)
- [ ] Smart contract verification (Solidity, Move, Rust)
- [ ] Hardware design verification (Chisel, Verilog)

### 11. Ecosystem
- [ ] VS Code extension for schema authoring
- [ ] Language server protocol (LSP) for Veritas DSL
- [ ] Package registry for reusable schemas/cartridges
- [ ] Cloud-native deployment (Kubernetes operator)

---

## Maintenance

### 12. Technical Debt
- [ ] Remove `UnsupportedPattern` fallbacks where full support added
- [ ] Consolidate duplicate interval logic between Rust/Python
- [ ] Add more property-based tests (proptest/quickcheck)

### 13. Testing
- [ ] Fuzzing for JSON PDA and contractor
- [ ] Differential testing against Z3 for linear arithmetic
- [ ] Integration tests with real LLM providers (vLLM, SGLang, llama.cpp)

---

## Priority Order
1. **Documentation** - Unblocks users
2. **Native parity** - Unblocks accelerated deployment
3. **Quadratic improvements** - High-value for quantitative use cases
4. **GBNF enhancements** - Unblocks schema-enforced llama.cpp
5. **Certificate chains** - Production security requirement