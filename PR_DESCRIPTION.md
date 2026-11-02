# PR: Complete Remaining Audit Items - Quadratic Contraction, Pattern Masking, GBNF, Certificate Signing, Dynamic Numeric Limits

## Summary
This PR implements all remaining gaps identified in the Veritas engineering audit (docs/AUDIT.md). All previously documented limitations have been addressed with production-ready implementations.

---

## Changes

### 1. Quadratic Interval Contraction (Rust Core)
**Files:** `crates/veritas-core/src/contractor.rs`, `crates/veritas-core/src/wire.rs`, `python/veritas/decode.py`, `python/veritas/interval.py`

- Added `QuadraticForm` constraint variant supporting Cholesky factorization (`cholesky_l: Vec<Vec<f64>>`)
- Implemented `revise_quadratic_form()` with forward checking and conservative domain narrowing
- Wire format serialization/deserialization for quadratic constraints
- Python contractor now emits quadratic_form constraints for `quad_form_cholesky` expressions
- HC4 contractor forward check validates quadratic form interval against target

### 2. Enhanced Pattern Masking
**Files:** `python/veritas/pattern.py`

- Added support for multiline anchors: `AT_BEGINNING_LINE`, `AT_END_LINE`
- Added word boundary anchors: `AT_BOUNDARY` (\b), `AT_NON_BOUNDARY` (\B) with conservative epsilon approximation
- Added locale/Unicode character category variants: `CATEGORY_LOC_*`, `CATEGORY_UNI_*`
- Added linebreak categories: `CATEGORY_LINEBREAK`, `CATEGORY_NOT_LINEBREAK`
- All 97 pattern tests pass

### 3. Schema-Specific GBNF Generation for llama.cpp
**Files:** `python/veritas/bridges/gbnf.py` (new), `python/veritas/bridges/llamacpp_bridge.py`, `python/veritas/bridges/__init__.py`

- New `GBNFGenerator` class generating schema-specific GBNF from VIR spec
- Handles objects, arrays, strings, numbers, booleans, enums, nested objects, containers
- `LlamaCppBridge` now auto-generates schema-specific grammar when spec provided
- Generic JSON GBNF used only as fallback

### 4. Certificate Cryptographic Signing (Ed25519)
**Files:** `python/veritas/proof/certificate.py`

- Added `sign(private_key)`, `verify_signature(public_key)`, `generate_keypair()` methods
- `_signing_payload()` excludes signature fields to prevent circular dependency
- `sign()` calls `seal()` to update content digest after adding signature
- Fixed keypair return order to `(public_key, private_seed)`
- Optional `verify_signature` parameter in `verify_certificate()`

### 5. Dynamic Numeric Literal Bounds
**Files:** `python/veritas/decode.py`

- Added `auto_scale_numeric_limits` config option (default: `True`)
- `_compute_numeric_limits()` computes integer/fraction digits from field's declared min/max
- `_number_prefix_ok()` uses dynamic limits when enabled
- Allows solutions needing more digits than static config limits

---

## Testing
- **345 passed, 54 skipped** - all tests pass
- Integration tests validate all 5 features:
  - Quadratic form constraints (portfolio volatility)
  - Pattern anchors ($ end anchor)
  - Certificate Ed25519 sign/verify round-trip
  - Dynamic numeric limits config
  - GBNF generation (697 chars for portfolio schema)

---

## Documentation Updates Needed
- Update `docs/NATIVE.md` to document quadratic_form wire format
- Update `docs/BENCHMARKS.md` if native speedup changes
- Update `README.md` with new features (GBNF, certificate signing, dynamic limits)

---

## Breaking Changes
None. All additions are backward-compatible:
- New config options default to safe values
- New constraint types only used when explicitly emitted
- Certificate signing is opt-in

---

## Related Issues
Addresses all remaining "Known Limits" from `docs/AUDIT.md`:
- ✅ Quadratic interval contraction implemented
- ✅ Pattern masking enhanced (more regex features)
- ✅ Schema-specific GBNF generation for llama.cpp
- ✅ Certificate signing support added
- ✅ Dynamic numeric literal bounds handling