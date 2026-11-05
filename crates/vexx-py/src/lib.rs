//! `vexx_py` — the CPython extension module (abi3, pyo3).
//!
//! Every entry point is a pure function of its arguments (the automaton is exposed through
//! [`PdaHandle`], one handle per decoding stream). That is what makes the native path safe to
//! prefer: there is no global mutable state to corrupt, and the Python engine can fall back
//! to pure Python at any moment.
//!
//! The JSON-in/JSON-out shapes are documented in `vexx_core::wire`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use vexx_core::contractor::Contractor;
use vexx_core::interval::Interval;
use vexx_core::json::{self, Json, Prefix, VocabularyTrie};
use vexx_core::num::Rational;
use vexx_core::pda::Pda;
use vexx_core::solver::{self, Linear, Outcome};
use vexx_core::wire;

fn parse_json(text: &str) -> PyResult<Json> {
    json::parse(text).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn status_name(status: Prefix) -> &'static str {
    match status {
        Prefix::Complete => "complete",
        Prefix::Incomplete => "incomplete",
        Prefix::Invalid => "invalid",
    }
}

/// `"complete"`, `"incomplete"` or `"invalid"` for a partial JSON document.
#[pyfunction]
fn prefix_status(text: &str) -> &'static str {
    status_name(json::prefix_status(text))
}

#[pyfunction]
fn validate_document(text: &str) -> bool {
    json::prefix_status(text) == Prefix::Complete && json::parse(text).is_ok()
}

/// Canonical re-rendering (`json.dumps(value, sort_keys=False, separators=(", ", ": "))`).
#[pyfunction]
fn canonical_document(text: &str) -> PyResult<Option<String>> {
    Ok(json::parse(text).ok().map(|value| value.render()))
}

#[pyfunction]
fn document_value(text: &str) -> PyResult<String> {
    let value = parse_json(text)?;
    Ok(value.render())
}

#[pyfunction]
#[pyo3(signature = (prefix, vocabulary, eos=None))]
fn allowed_tokens(prefix: &str, vocabulary: Vec<String>, eos: Option<String>) -> Vec<u32> {
    json::mask(prefix, &vocabulary, eos.as_deref())
}

/// Vocabulary trie for accelerated token masking.
/// Build once per tokenizer vocabulary, then reuse for fast mask queries.
#[pyclass]
struct PyVocabularyTrie {
    inner: VocabularyTrie,
}

#[pymethods]
impl PyVocabularyTrie {
    #[new]
    #[pyo3(signature = (vocabulary, eos=None))]
    fn new(vocabulary: Vec<String>, eos: Option<String>) -> PyResult<Self> {
        if vocabulary.is_empty() {
            return Err(PyValueError::new_err("vocabulary must not be empty"));
        }
        let trie = VocabularyTrie::new(&vocabulary, eos.as_deref())
            .ok_or_else(|| PyValueError::new_err("failed to build vocabulary trie"))?;
        Ok(PyVocabularyTrie { inner: trie })
    }
    
    /// Get all token indices that have `prefix` as a prefix.
    fn tokens_with_prefix(&self, prefix: &str) -> Vec<u32> {
        self.inner.tokens_with_prefix(prefix)
    }

    /// Fast mask using the trie.  The vocabulary is owned by the trie, so callers no longer
    /// clone 128k strings on every decoding step.
    #[pyo3(signature = (prefix, eos=None))]
    fn mask(&self, prefix: &str, eos: Option<String>) -> Vec<u32> {
        self.inner.mask(prefix, eos.as_deref())
    }

    /// Packed `u64` validity bitset over the full vocabulary (`ceil(|V| / 64)` words):
    /// masks a logits row with one vectorized AND instead of per-index gathers.
    #[pyo3(signature = (prefix, eos=None))]
    fn mask_bitset(&self, prefix: &str, eos: Option<String>) -> Vec<u64> {
        self.inner.mask_bitset(prefix, eos.as_deref())
    }

    /// Number of `u64` words `mask_bitset` returns.
    fn bitset_words(&self) -> usize {
        self.inner.bitset_words()
    }

    /// Size of the vocabulary the trie was built from.
    fn vocabulary_len(&self) -> usize {
        self.inner.vocabulary_len()
    }
}

#[pyfunction]
fn numeric_prefix(text: &str, lo: f64, hi: f64, allow_exp: bool) -> (f64, f64) {
    let span = Interval::from_numeric_prefix(text, Interval::new(lo, hi), allow_exp);
    if span.is_empty() {
        (f64::INFINITY, f64::NEG_INFINITY)
    } else {
        (span.lo, span.hi)
    }
}

#[pyfunction]
#[pyo3(signature = (domains_json, constraints_json, rounds = 8))]
fn contract(domains_json: &str, constraints_json: &str, rounds: usize) -> PyResult<String> {
    let domains = parse_json(domains_json)?;
    let constraints = parse_json(constraints_json)?;
    let (contractor, refuted) = wire::run_contractor(&domains, &constraints, rounds)
        .map_err(PyValueError::new_err)?;
    Ok(wire::contractor_json(&contractor, refuted).render())
}

#[pyfunction]
fn log_volume(domains_json: &str) -> PyResult<Option<f64>> {
    let domains = parse_json(domains_json)?;
    let parsed = wire::parse_domains(&domains).map_err(PyValueError::new_err)?;
    Ok(Contractor::new(parsed).log_volume())
}

#[pyfunction]
fn solve_linear(linear_json: &str) -> PyResult<String> {
    let node = parse_json(linear_json)?;
    let constraints: Vec<Linear> = wire::parse_linear(&node).map_err(PyValueError::new_err)?;
    let outcome = solver::solve(&constraints);
    Ok(wire::outcome_json(&outcome).render())
}

/// Exact-rational alternative to [`solve_linear`], for callers that want `1/3` not `0.333…`.
#[pyfunction]
fn solve_linear_exact(linear_json: &str) -> PyResult<Vec<(u32, String)>> {
    let node = parse_json(linear_json)?;
    let constraints: Vec<Linear> = wire::parse_linear(&node).map_err(PyValueError::new_err)?;
    match solver::solve(&constraints) {
        Outcome::Sat { exact, .. } => Ok(exact
            .into_iter()
            .map(|(var, value): (u32, Rational)| (var, value.to_string()))
            .collect()),
        Outcome::Unsat { core } => Err(PyValueError::new_err(format!(
            "unsatisfiable (core {:?})",
            core
        ))),
        Outcome::Unknown { reason } => Err(PyValueError::new_err(reason)),
    }
}

/// One decoding stream: feed text, ask for the mask, clone for candidate lookahead.
#[pyclass]
struct PdaHandle {
    inner: Pda,
}

#[pymethods]
impl PdaHandle {
    #[new]
    fn new() -> PdaHandle {
        PdaHandle { inner: Pda::new() }
    }

    /// `False` once the stream can never become a valid document.
    fn push(&mut self, text: &str) -> bool {
        self.inner.push_str(text)
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    fn accepts(&self) -> bool {
        self.inner.accepts()
    }

    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn last_event(&self) -> Option<&'static str> {
        self.inner.last_event().map(|event| match event {
            vexx_core::pda::Event::Open => "open",
            vexx_core::pda::Event::Close => "close",
            vexx_core::pda::Event::Key => "key",
            vexx_core::pda::Event::Value => "value",
            vexx_core::pda::Event::Separator => "separator",
        })
    }

    fn candidates(&self, vocabulary: Vec<String>) -> Vec<u32> {
        self.inner.candidates(&vocabulary)
    }

    #[pyo3(signature = (vocabulary, eos=None))]
    fn mask(&self, vocabulary: Vec<String>, eos: Option<String>) -> Vec<u32> {
        self.inner.mask(&vocabulary, eos.as_deref())
    }

    fn clone_handle(&self) -> PdaHandle {
        PdaHandle {
            inner: self.inner.clone(),
        }
    }
}

/// Snapshot of the engine, used by `vexx.doctor()` to prove which binary is loaded.
#[pyfunction]
fn engine_info() -> String {
    Json::Object(vec![
        ("name".to_string(), Json::Str("vexx_py".to_string())),
        (
            "version".to_string(),
            Json::Str(vexx_core::version().to_string()),
        ),
        (
            "wire_version".to_string(),
            Json::Number(vexx_core::wire_version() as f64),
        ),
        ("abi".to_string(), Json::Str("abi3-py311".to_string())),
    ])
    .render()
}

#[pymodule]
fn vexx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", vexx_core::version())?;
    m.add("WIRE_VERSION", vexx_core::wire_version())?;
    m.add_function(wrap_pyfunction!(prefix_status, m)?)?;
    m.add_function(wrap_pyfunction!(validate_document, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_document, m)?)?;
    m.add_function(wrap_pyfunction!(document_value, m)?)?;
    m.add_function(wrap_pyfunction!(allowed_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(numeric_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(contract, m)?)?;
    m.add_function(wrap_pyfunction!(log_volume, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear, m)?)?;
    m.add_function(wrap_pyfunction!(solve_linear_exact, m)?)?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_class::<PdaHandle>()?;
    m.add_class::<PyVocabularyTrie>()?;
    Ok(())
}
