//! Strict JSON parsing, prefix recognition and vocabulary masking (Tier 0).
//!
//! The recognizer answers three questions the decoder needs:
//!
//! * [`parse`] — is this a complete, canonical JSON document? (RFC 8259: no `NaN`, no
//!   duplicate keys, no trailing data, bounded nesting.)
//! * [`prefix_status`] — is `text` a valid *prefix* of some document, i.e. may the model
//!   keep going (`Incomplete`), must it stop (`Complete`), or has it already failed
//!   (`Invalid`)? This is the exact mask contract.
//! * [`allowed_tokens`] — given a vocabulary, which tokens keep the document parseable?
//! * [`VocabularyTrie`] — a byte-trie built from the vocabulary for O(|prefix| + |valid|) masking.
//!
//! Duplicate-key rejection and the strict number grammar match `python/veritas/json_pda.py`;
//! the mask is therefore the same language, and the Python side stays authoritative for
//! anything this module reports as [`Prefix::Invalid`] *and* the native mask is used only as
//! an accelerator (see `python/veritas/native.py`).

use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(value) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Vec<(String, Json)>> {
        match self {
            Json::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// Deterministic rendering with the same spelling as `python/veritas/json_pda.py`.
    pub fn render(&self) -> String {
        match self {
            Json::Null => "null".to_string(),
            Json::Bool(true) => "true".to_string(),
            Json::Bool(false) => "false".to_string(),
            Json::Number(value) => render_number(*value),
            Json::Str(value) => render_string(value),
            Json::Array(items) => {
                let parts: Vec<String> = items.iter().map(|item| item.render()).collect();
                format!("[{}]", parts.join(", "))
            }
            Json::Object(entries) => {
                let parts: Vec<String> = entries
                    .iter()
                    .map(|(key, value)| format!("{}: {}", render_string(key), value.render()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }
}

/// `json_number(value, precision)` from `python/veritas/json_pda.py`: round to `precision`
/// decimals and render the shortest round-tripping form, integers without a decimal point.
///
/// Rounding is half-away-from-zero, where Python's `round` is half-to-even; the difference
/// only shows up on exact ties, and the splice text is advisory (the value itself is exact).
pub fn json_number(value: f64, precision: Option<u32>) -> String {
    if !value.is_finite() {
        return "null".to_string();
    }
    let rounded = match precision {
        Some(digits) => round_to(value, digits),
        None => value,
    };
    if rounded == rounded.trunc() && rounded.abs() < 1e16 {
        return format!("{}", rounded as i64);
    }
    format!("{:?}", rounded)
}

pub fn round_to(value: f64, digits: u32) -> f64 {
    let factor = 10f64.powi(digits as i32);
    if !factor.is_finite() || factor == 0.0 {
        return value;
    }
    (value * factor).round() / factor
}

pub fn render_number(value: f64) -> String {
    if !value.is_finite() {
        return "null".to_string();
    }
    if value == value.trunc() && value.abs() < 1e16 {
        return format!("{}", value as i64);
    }
    format!("{:?}", value)
}

pub fn render_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct JsonError {
    pub offset: usize,
    pub detail: String,
}

impl JsonError {
    fn new(offset: usize, detail: &str) -> JsonError {
        JsonError {
            offset,
            detail: detail.to_string(),
        }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at offset {}", self.detail, self.offset)
    }
}

/// Tri-state verdict for a prefix of a JSON document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefix {
    /// A complete document: stopping is allowed, nothing may follow.
    Complete,
    /// A valid start of some document: more input is required (or allowed).
    Incomplete,
    /// Not a prefix of any document.
    Invalid,
}

const MAX_DEPTH: usize = 64;

pub fn parse(text: &str) -> Result<Json, JsonError> {
    let bytes = text.as_bytes();
    let mut index = skip_ws(bytes, 0);
    let (value, next) = parse_value(bytes, index, 0, false)?;
    let value = match value {
        Some(value) => value,
        None => return Err(JsonError::new(next, "unexpected end of input")),
    };
    index = skip_ws(bytes, next);
    if index != bytes.len() {
        return Err(JsonError::new(index, "trailing data after document"));
    }
    Ok(value)
}

pub fn prefix_status(text: &str) -> Prefix {
    let bytes = text.as_bytes();
    let start = skip_ws(bytes, 0);
    if start == bytes.len() {
        return Prefix::Incomplete;
    }
    match parse_value(bytes, start, 0, true) {
        Err(_) => Prefix::Invalid,
        Ok((value, next)) => {
            if value.is_none() {
                return Prefix::Incomplete;
            }
            let trailing = skip_ws(bytes, next);
            if trailing == bytes.len() {
                Prefix::Complete
            } else {
                Prefix::Invalid
            }
        }
    }
}

/// Number of tokens from `vocabulary` that keep `prefix` parseable. The special token
/// `"<stop>"` (index reported in the same order) is allowed only for a complete document.
pub fn allowed_tokens(prefix: &str, vocabulary: &[String]) -> Vec<u32> {
    let mut allowed: Vec<u32> = Vec::new();
    for (index, token) in vocabulary.iter().enumerate() {
        let mut candidate = String::with_capacity(prefix.len() + token.len());
        candidate.push_str(prefix);
        candidate.push_str(token);
        if prefix_status(&candidate) != Prefix::Invalid {
            allowed.push(index as u32);
        }
    }
    allowed
}

/// Exact token mask including the end-of-sequence token, mirroring `pda.py`.
pub fn mask(prefix: &str, vocabulary: &[String], eos: Option<&str>) -> Vec<u32> {
    let mut allowed = allowed_tokens(prefix, vocabulary);
    if let Some(eos_token) = eos {
        if prefix_status(prefix) == Prefix::Complete {
            for (index, token) in vocabulary.iter().enumerate() {
                if token == eos_token {
                    allowed.push(index as u32);
                }
            }
        }
    }
    allowed.sort_unstable();
    allowed.dedup();
    allowed
}

/// A character trie for fast vocabulary masking, driven by the JSON PDA.
///
/// The trie is built once per tokenizer vocabulary.  Masking performs a depth-first
/// traversal: at every node the machine is cloned and advanced by one character; branches
/// that die are pruned in O(1), and a branch that reaches a full token records its index.
/// The result is computed in O(sum of token prefix lengths that stay alive) — typically a
/// few hundred node visits, instead of O(|vocabulary|) document re-parses.  Multi-byte
/// UTF-8 characters are consumed atomically, so no partial code-point can be emitted.
///
/// Empty tokens are ignored (they make no progress).
#[derive(Clone, Debug)]
pub struct VocabularyTrie {
    /// Tree nodes; root (index 0) is "no input consumed".
    nodes: Vec<TrieNode>,
    /// Token strings, stored to answer EOS handling and diagnostics.
    vocabulary: Vec<String>,
    /// Index of the EOS token in `vocabulary`, when known.
    eos_token: Option<usize>,
}

/// A node in the vocabulary trie.
#[derive(Clone, Debug, Default)]
struct TrieNode {
    /// Children keyed by character; `BTreeMap` iterates keys in order for determinism.
    children: std::collections::BTreeMap<char, usize>,
    /// Token indices terminating exactly at this node.
    token_indices: Vec<u32>,
}

impl VocabularyTrie {
    /// Build the trie from a vocabulary.  Returns `None` for an empty vocabulary.
    pub fn new(vocabulary: &[String], eos_token: Option<&str>) -> Option<Self> {
        if vocabulary.is_empty() {
            return None;
        }
        let mut trie = VocabularyTrie {
            nodes: vec![TrieNode::default()],
            vocabulary: vocabulary.to_vec(),
            eos_token: None,
        };
        for (index, token) in vocabulary.iter().enumerate() {
            let mut node_idx = 0usize;
            for ch in token.chars() {
                let next = match trie.nodes[node_idx].children.get(&ch) {
                    Some(&child) => child,
                    None => {
                        let child = trie.nodes.len();
                        trie.nodes.push(TrieNode::default());
                        trie.nodes[node_idx].children.insert(ch, child);
                        child
                    }
                };
                node_idx = next;
            }
            trie.nodes[node_idx].token_indices.push(index as u32);
        }
        trie.eos_token = eos_token
            .and_then(|eos| vocabulary.iter().position(|t| t == eos));
        Some(trie)
    }

    /// `Token indices with `prefix` as a literal prefix (unsorted, with duplicates removed).
    pub fn tokens_with_prefix(&self, prefix: &str) -> Vec<u32> {
        let mut node_idx = 0usize;
        for ch in prefix.chars() {
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => node_idx = next,
                None => return Vec::new(),
            }
        }
        self.collect_token_indices(node_idx)
    }

    /// All token indices in the subtree rooted at `node_idx`, sorted and deduplicated.
    fn collect_token_indices(&self, node_idx: usize) -> Vec<u32> {
        let mut result = Vec::new();
        let mut stack = vec![node_idx];
        while let Some(idx) = stack.pop() {
            result.extend_from_slice(&self.nodes[idx].token_indices);
            stack.extend(self.nodes[idx].children.values().copied());
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Indices of the tokens that keep the document alive after `prefix`.
    ///
    /// Equivalent to [`super::json::mask`] with respect to this crate's PDA; empty tokens
    /// are never reported.  The end-of-sequence token is included only when the document is
    /// complete.
    pub fn mask(&self, prefix: &str, eos: Option<&str>) -> Vec<u32> {
        let mut base = crate::pda::Pda::new();
        if !base.push_str(prefix) {
            return Vec::new();
        }
        let mut allowed = Vec::new();
        self.dfs_mask(0, &base, &mut allowed);
        self.append_eos(&base, eos, &mut allowed);
        allowed.sort_unstable();
        allowed.dedup();
        allowed
    }

    /// Packed validity bitset: word `i / 64`, bit `i % 64` set iff token `i` is allowed.
    ///
    /// Length is `ceil(|vocabulary| / 64)`; callers map it onto logits with one
    /// vectorized mask multiply instead of per-index gathers.
    pub fn mask_bitset(&self, prefix: &str, eos: Option<&str>) -> Vec<u64> {
        let allowed = self.mask(prefix, eos);
        self.bitset_from_allowed(&allowed)
    }

    fn bitset_from_allowed(&self, allowed: &[u32]) -> Vec<u64> {
        let words = self.vocabulary.len().div_ceil(64);
        let mut bits = vec![0u64; words];
        for &index in allowed {
            let index = index as usize;
            if index / 64 < words {
                bits[index / 64] |= 1u64 << (index % 64);
            }
        }
        bits
    }

    /// Number of words `mask_bitset` returns (`ceil(|vocabulary| / 64)`).
    pub fn bitset_words(&self) -> usize {
        self.vocabulary.len().div_ceil(64)
    }

    fn dfs_mask(&self, node_idx: usize, pda: &crate::pda::Pda, allowed: &mut Vec<u32>) {
        let node = &self.nodes[node_idx];
        if node_idx > 0 {
            allowed.extend_from_slice(&node.token_indices);
        }
        for (&ch, &child_idx) in &node.children {
            let mut next = pda.clone();
            if next.step(ch) && next.is_alive() {
                self.dfs_mask(child_idx, &next, allowed);
            }
        }
    }

    fn append_eos(&self, base: &crate::pda::Pda, eos: Option<&str>, allowed: &mut Vec<u32>) {
        let Some(eos) = eos else { return };
        if !base.is_complete() {
            return;
        }
        match self.eos_token {
            Some(index) if self.vocabulary[index] == eos => allowed.push(index as u32),
            _ => {
                for (index, token) in self.vocabulary.iter().enumerate() {
                    if token == eos {
                        allowed.push(index as u32);
                    }
                }
            }
        }
    }
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() {
        match bytes[index] {
            b' ' | b'\t' | b'\n' | b'\r' => index += 1,
            _ => break,
        }
    }
    index
}

/// Parse one value starting at `index`.
///
/// Returns `(Some(value), next_index)` for a finished value, `(None, index)` when the input
/// ended inside the value (only possible with `tolerant = true`), or an error for input that
/// cannot become valid JSON.
fn parse_value(
    bytes: &[u8],
    index: usize,
    depth: usize,
    tolerant: bool,
) -> Result<(Option<Json>, usize), JsonError> {
    if depth > MAX_DEPTH {
        return Err(JsonError::new(index, "maximum nesting depth exceeded"));
    }
    let start = skip_ws(bytes, index);
    if start >= bytes.len() {
        if tolerant {
            return Ok((None, start));
        }
        return Err(JsonError::new(start, "unexpected end of input"));
    }
    match bytes[start] {
        b'{' => parse_object(bytes, start + 1, depth + 1, tolerant),
        b'[' => parse_array(bytes, start + 1, depth + 1, tolerant),
        b'"' => {
            let (value, next) = parse_string(bytes, start + 1, tolerant)?;
            match value {
                Some(text) => Ok((Some(Json::Str(text)), next)),
                None => Ok((None, next)),
            }
        }
        b't' => parse_literal(bytes, start, "true", Json::Bool(true), tolerant),
        b'f' => parse_literal(bytes, start, "false", Json::Bool(false), tolerant),
        b'n' => parse_literal(bytes, start, "null", Json::Null, tolerant),
        b'-' | b'0'..=b'9' => parse_number(bytes, start, tolerant),
        _ => Err(JsonError::new(start, "unexpected character")),
    }
}

fn parse_literal(
    bytes: &[u8],
    index: usize,
    word: &str,
    value: Json,
    tolerant: bool,
) -> Result<(Option<Json>, usize), JsonError> {
    let word_bytes = word.as_bytes();
    let mut position = 0usize;
    while position < word_bytes.len() {
        let at = index + position;
        if at >= bytes.len() {
            if tolerant {
                return Ok((None, at));
            }
            return Err(JsonError::new(at, "unexpected end of input"));
        }
        if bytes[at] != word_bytes[position] {
            return Err(JsonError::new(at, "invalid literal"));
        }
        position += 1;
    }
    Ok((Some(value), index + word_bytes.len()))
}

fn parse_number(
    bytes: &[u8],
    index: usize,
    tolerant: bool,
) -> Result<(Option<Json>, usize), JsonError> {
    let mut at = index;
    if at < bytes.len() && bytes[at] == b'-' {
        at += 1;
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
    }
    // Integer part: 0 | [1-9][0-9]*
    if bytes[at] == b'0' {
        at += 1;
    } else if bytes[at].is_ascii_digit() {
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
    } else {
        return Err(JsonError::new(at, "invalid number"));
    }
    // Fraction
    if at < bytes.len() && bytes[at] == b'.' {
        at += 1;
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        if !bytes[at].is_ascii_digit() {
            return Err(JsonError::new(at, "invalid fraction"));
        }
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
    }
    // Exponent
    if at < bytes.len() && (bytes[at] == b'e' || bytes[at] == b'E') {
        at += 1;
        if at < bytes.len() && (bytes[at] == b'+' || bytes[at] == b'-') {
            at += 1;
        }
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        if !bytes[at].is_ascii_digit() {
            return Err(JsonError::new(at, "invalid exponent"));
        }
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
    }
    let text = std::str::from_utf8(&bytes[index..at])
        .map_err(|_| JsonError::new(index, "number is not valid UTF-8"))?;
    let value: f64 = text
        .parse()
        .map_err(|_| JsonError::new(index, "number out of range"))?;
    if !value.is_finite() {
        return Err(JsonError::new(index, "number is not finite"));
    }
    // A trailing token character means the number was truncated by the mask, not finished.
    if at < bytes.len() {
        let next = bytes[at];
        let boundary = next.is_ascii_whitespace()
            || next == b','
            || next == b']'
            || next == b'}'
            || next == b':';
        if !boundary {
            return Err(JsonError::new(at, "invalid character after number"));
        }
    }
    Ok((Some(Json::Number(value)), at))
}

fn parse_string(
    bytes: &[u8],
    index: usize,
    tolerant: bool,
) -> Result<(Option<String>, usize), JsonError> {
    let mut out = String::new();
    let mut at = index;
    while at < bytes.len() {
        let byte = bytes[at];
        match byte {
            b'"' => return Ok((Some(out), at + 1)),
            b'\\' => {
                at += 1;
                if at >= bytes.len() {
                    return if tolerant {
                        Ok((None, at))
                    } else {
                        Err(JsonError::new(at, "unexpected end of input"))
                    };
                }
                match bytes[at] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let (ch, next) = parse_unicode_escape(bytes, at + 1, tolerant)?;
                        match ch {
                            Some(ch) => out.push(ch),
                            None => return Ok((None, next)),
                        }
                        at = next;
                        continue;
                    }
                    _ => return Err(JsonError::new(at, "invalid escape")),
                }
                at += 1;
            }
            byte if byte < 0x20 => {
                return Err(JsonError::new(at, "unescaped control character"));
            }
            byte if byte < 0x80 => {
                out.push(byte as char);
                at += 1;
            }
            _ => {
                // Multi-byte UTF-8: copy the whole sequence.
                let width = utf8_width(byte);
                if width == 0 || at + width > bytes.len() {
                    return if tolerant && at + width > bytes.len() {
                        Ok((None, bytes.len()))
                    } else {
                        Err(JsonError::new(at, "invalid UTF-8"))
                    };
                }
                let chunk = std::str::from_utf8(&bytes[at..at + width])
                    .map_err(|_| JsonError::new(at, "invalid UTF-8"))?;
                out.push_str(chunk);
                at += width;
            }
        }
    }
    if tolerant {
        Ok((None, at))
    } else {
        Err(JsonError::new(at, "unterminated string"))
    }
}

fn utf8_width(first: u8) -> usize {
    if first & 0b1110_0000 == 0b1100_0000 {
        2
    } else if first & 0b1111_0000 == 0b1110_0000 {
        3
    } else if first & 0b1111_1000 == 0b1111_0000 {
        4
    } else {
        0
    }
}

fn parse_unicode_escape(
    bytes: &[u8],
    index: usize,
    tolerant: bool,
) -> Result<(Option<char>, usize), JsonError> {
    let mut value: u32 = 0;
    let mut at = index;
    for _ in 0..4 {
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        let digit = match bytes[at] {
            b @ b'0'..=b'9' => (b - b'0') as u32,
            b @ b'a'..=b'f' => (b - b'a' + 10) as u32,
            b @ b'A'..=b'F' => (b - b'A' + 10) as u32,
            _ => return Err(JsonError::new(at, "invalid unicode escape")),
        };
        value = value * 16 + digit;
        at += 1;
    }
    let ch = char::from_u32(value).ok_or_else(|| JsonError::new(at, "invalid code point"))?;
    Ok((Some(ch), at))
}

fn parse_array(
    bytes: &[u8],
    index: usize,
    depth: usize,
    tolerant: bool,
) -> Result<(Option<Json>, usize), JsonError> {
    let mut items: Vec<Json> = Vec::new();
    let mut at = skip_ws(bytes, index);
    if at >= bytes.len() {
        return if tolerant {
            Ok((None, at))
        } else {
            Err(JsonError::new(at, "unexpected end of input"))
        };
    }
    if bytes[at] == b']' {
        return Ok((Some(Json::Array(items)), at + 1));
    }
    loop {
        let (value, next) = parse_value(bytes, at, depth, tolerant)?;
        let value = match value {
            Some(value) => value,
            None => return Ok((None, next)),
        };
        items.push(value);
        at = skip_ws(bytes, next);
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        match bytes[at] {
            b',' => {
                at = skip_ws(bytes, at + 1);
                if at >= bytes.len() {
                    return if tolerant {
                        Ok((None, at))
                    } else {
                        Err(JsonError::new(at, "unexpected end of input"))
                    };
                }
            }
            b']' => return Ok((Some(Json::Array(items)), at + 1)),
            _ => return Err(JsonError::new(at, "expected ',' or ']'")),
        }
    }
}

fn parse_object(
    bytes: &[u8],
    index: usize,
    depth: usize,
    tolerant: bool,
) -> Result<(Option<Json>, usize), JsonError> {
    let mut entries: Vec<(String, Json)> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut at = skip_ws(bytes, index);
    if at >= bytes.len() {
        return if tolerant {
            Ok((None, at))
        } else {
            Err(JsonError::new(at, "unexpected end of input"))
        };
    }
    if bytes[at] == b'}' {
        return Ok((Some(Json::Object(entries)), at + 1));
    }
    loop {
        at = skip_ws(bytes, at);
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        if bytes[at] != b'"' {
            return Err(JsonError::new(at, "expected string key"));
        }
        let (key, next) = parse_string(bytes, at + 1, tolerant)?;
        let key = match key {
            Some(key) => key,
            None => return Ok((None, next)),
        };
        if !seen.insert(key.clone()) {
            return Err(JsonError::new(at, "duplicate object key"));
        }
        at = skip_ws(bytes, next);
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        if bytes[at] != b':' {
            return Err(JsonError::new(at, "expected ':'"));
        }
        let (value, next) = parse_value(bytes, at + 1, depth, tolerant)?;
        let value = match value {
            Some(value) => value,
            None => return Ok((None, next)),
        };
        entries.push((key, value));
        at = skip_ws(bytes, next);
        if at >= bytes.len() {
            return if tolerant {
                Ok((None, at))
            } else {
                Err(JsonError::new(at, "unexpected end of input"))
            };
        }
        match bytes[at] {
            b',' => {
                at = skip_ws(bytes, at + 1);
                if at >= bytes.len() {
                    return if tolerant {
                        Ok((None, at))
                    } else {
                        Err(JsonError::new(at, "unexpected end of input"))
                    };
                }
            }
            b'}' => return Ok((Some(Json::Object(entries)), at + 1)),
            _ => return Err(JsonError::new(at, "expected ',' or '}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_documents() {
        let value = parse("{\"a\": 1, \"b\": [true, null, \"x\"]}").expect("valid");
        assert_eq!(value.get("a").and_then(|v| v.as_f64()), Some(1.0));
        assert_eq!(value.render(), "{\"a\": 1, \"b\": [true, null, \"x\"]}");
        assert!(parse("1e3").is_ok());
        assert!(parse("-0.5").is_ok());
        assert!(parse("\"\\u00e9\"").is_ok());
    }

    #[test]
    fn rejects_non_canonical_input() {
        assert!(parse("{\"a\": 1, \"a\": 2}").is_err());
        assert!(parse("01").is_err());
        assert!(parse("NaN").is_err());
        assert!(parse("{\"a\": 1,}").is_err());
        assert!(parse("1 2").is_err());
        assert!(parse("[1,").is_err());
        assert!(parse("\"unterminated").is_err());
    }

    #[test]
    fn prefix_verdicts_match_the_mask_contract() {
        assert_eq!(prefix_status(""), Prefix::Incomplete);
        assert_eq!(prefix_status("{"), Prefix::Incomplete);
        assert_eq!(prefix_status("{\"a\": "), Prefix::Incomplete);
        assert_eq!(prefix_status("{\"a\": 1"), Prefix::Incomplete);
        assert_eq!(prefix_status("{\"a\": 1}"), Prefix::Complete);
        assert_eq!(prefix_status("{\"a\": 1, \"b\": [2, 3]}"), Prefix::Complete);
        assert_eq!(prefix_status("{\"a\": 1} "), Prefix::Complete);
        assert_eq!(prefix_status("{\"a\" 1}"), Prefix::Invalid);
        assert_eq!(prefix_status("[1,]"), Prefix::Invalid);
        assert_eq!(prefix_status("nul"), Prefix::Incomplete);
        assert_eq!(prefix_status("nullx"), Prefix::Invalid);
        assert_eq!(prefix_status("tru"), Prefix::Incomplete);
    }

    #[test]
    fn masks_only_parseable_tokens() {
        let vocabulary: Vec<String> = vec![
            "{".to_string(),
            "}".to_string(),
            "\"a\"".to_string(),
            ": ".to_string(),
            "1".to_string(),
            "x".to_string(),
            "<stop>".to_string(),
        ];
        let allowed = mask("", &vocabulary, Some("<stop>"));
        assert!(allowed.contains(&0)); // "{"
        assert!(!allowed.contains(&1)); // "}" alone is not a document prefix
        assert!(!allowed.contains(&5)); // "x" cannot start a value
        assert!(!allowed.contains(&6)); // stop is not allowed yet
        let done = mask("{\"a\": 1}", &vocabulary, Some("<stop>"));
        assert!(done.contains(&6));
        assert_eq!(done, vec![6]);
    }
}
