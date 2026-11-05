//! `vexx-core` — the dependency-free Rust engine behind `vexx`.
//!
//! The crate is deliberately dependency-free: every wheel ships from this source tree with no
//! C toolchain, no `cc` crate and no vendored solver. The pieces are
//!
//! | module | role |
//! |---|---|
//! | [`num`] | ULP stepping and exact `i128` rationals |
//! | [`interval`] | outward-rounded interval arithmetic + numeric prefix masks |
//! | [`json`] | strict RFC 8259 parsing, prefix recognition, vocabulary masks |
//! | [`pda`] | incremental pushdown automaton for masked decoding |
//! | [`expr`] | tri-state evaluation of the VIR expression wire format |
//! | [`contractor`] | HC4 contractor (linear, sum, square, product) |
//! | [`solver`] | exact-rational Fourier–Motzkin feasibility with cores |
//! | [`wire`] | the JSON wire format shared with Python and wasm |
//!
//! Everything that cannot be decided returns "unknown"; nothing here guesses. The Python
//! engine (`python/vexx/native.py`) treats the native results as an accelerator and keeps
//! the pure-Python path authoritative, so a bug here degrades speed, not soundness.

pub mod contractor;
pub mod expr;
pub mod interval;
pub mod json;
pub mod num;
pub mod pda;
pub mod solver;
pub mod wire;

// Engine modules: each is a port of the matching `python/vexx/*.py` module.
pub mod cdcl_jumpback;
pub mod crane_mode;
pub mod nano_solver;
pub mod smt_bridge;
pub mod token_healer;
pub mod zero_dof;

/// Semantic version of the engine, shared by the Python and wasm bindings.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Wire format revision; exported so Python can refuse a mismatched binary.
pub fn wire_version() -> u32 {
    wire::WIRE_VERSION
}
