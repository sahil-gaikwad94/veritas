//! `veritas-wasm` — the browser-side inspector.
//!
//! The wasm build exposes the same automaton as the Python extension, with one handle per
//! stream and no global mutable state, so the HTML inspector can drive two decoders at once
//! (masked vs. unmasked) and compare them.

use wasm_bindgen::prelude::*;

use veritas_core::json::{self, Json, Prefix};
use veritas_core::pda::Pda;
use veritas_core::solver;

#[wasm_bindgen]
pub fn version() -> String {
    veritas_core::version().to_string()
}

#[wasm_bindgen]
pub fn prefix_status(text: &str) -> String {
    match json::prefix_status(text) {
        Prefix::Complete => "complete".to_string(),
        Prefix::Incomplete => "incomplete".to_string(),
        Prefix::Invalid => "invalid".to_string(),
    }
}

#[wasm_bindgen]
pub fn is_valid_document(text: &str) -> bool {
    json::parse(text).is_ok()
}

/// Canonical rendering of a document, or `undefined` when it is not one.
#[wasm_bindgen]
pub fn canonical_document(text: &str) -> Option<String> {
    json::parse(text).ok().map(|value| value.render())
}

#[wasm_bindgen]
pub fn numeric_prefix(text: &str, lo: f64, hi: f64, allow_exp: bool) -> Vec<f64> {
    let span =
        veritas_core::interval::Interval::from_numeric_prefix(text, veritas_core::interval::Interval::new(lo, hi), allow_exp);
    if span.is_empty() {
        vec![f64::INFINITY, f64::NEG_INFINITY]
    } else {
        vec![span.lo, span.hi]
    }
}

#[wasm_bindgen]
pub fn solve_linear_json(linear_json: &str) -> String {
    let parsed = match json::parse(linear_json) {
        Ok(node) => node,
        Err(error) => return failure(&error.to_string()),
    };
    let constraints = match veritas_core::wire::parse_linear(&parsed) {
        Ok(constraints) => constraints,
        Err(reason) => return failure(&reason),
    };
    veritas_core::wire::outcome_json(&solver::solve(&constraints)).render()
}

fn failure(reason: &str) -> String {
    Json::Object(vec![
        ("status".to_string(), Json::Str("error".to_string())),
        ("reason".to_string(), Json::Str(reason.to_string())),
    ])
    .render()
}

#[wasm_bindgen]
pub struct PdaHandle {
    inner: Pda,
}

impl Default for PdaHandle {
    fn default() -> PdaHandle {
        PdaHandle::new()
    }
}

#[wasm_bindgen]
impl PdaHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PdaHandle {
        PdaHandle { inner: Pda::new() }
    }

    pub fn push(&mut self, text: &str) -> bool {
        self.inner.push_str(text)
    }

    pub fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    pub fn accepts(&self) -> bool {
        self.inner.accepts()
    }

    pub fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Mask as a JSON array of token indices.
    pub fn mask_json(&self, vocabulary_json: &str, eos: &str) -> String {
        let vocabulary = match json::parse(vocabulary_json) {
            Ok(node) => node,
            Err(error) => return failure(&error.to_string()),
        };
        let tokens: Vec<String> = match vocabulary.as_array() {
            Some(items) => items
                .iter()
                .filter_map(|item| item.as_str().map(|text| text.to_string()))
                .collect(),
            None => return failure("vocabulary must be a JSON array of strings"),
        };
        let eos = if eos.is_empty() { None } else { Some(eos) };
        let allowed = self.inner.mask(&tokens, eos);
        Json::Array(allowed.into_iter().map(|index| Json::Number(index as f64)).collect()).render()
    }
}
