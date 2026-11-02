//! Incremental JSON pushdown automaton for masked decoding.
//!
//! The decoder never re-parses its output: it carries this automaton, clones it per
//! candidate token, and keeps the candidates that leave the machine alive. That is what makes
//! format masking O(vocabulary) instead of O(document) per step.
//!
//! The machine is the streaming view of the language in [`crate::json`]. It deliberately does
//! *not* remember the set of object keys it has seen (a streaming recognizer cannot do that in
//! finite state); a document with duplicate keys is still rejected by [`crate::json::parse`],
//! which is the authoritative final check. Validity of prefixes matches
//! [`crate::json::prefix_status`], and the tests pin that equivalence.

/// Structural expectation for the next character.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    /// A value must start here (after `[`, `,` or `:`).
    Value,
    /// A value may start, or the just-opened container may close.
    ValueOrClose,
    /// A key must start here (after `,` in an object).
    Key,
    /// A key may start, or the just-opened object may close.
    KeyOrClose,
    Colon,
    Comma,
    /// A complete document; only whitespace may follow.
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Array,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Key,
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Structure,
    String { role: Role },
    Escape { role: Role },
    Unicode { role: Role, left: u8 },
    Number,
    Literal { word: &'static str, matched: usize },
    Dead,
}

/// Something the automaton did, in order. The decoder uses this to know whether it just
/// closed an object (and therefore what may follow a separator).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Open,
    Close,
    Key,
    Value,
    Separator,
}

#[derive(Clone, Debug)]
pub struct Pda {
    stack: Vec<Frame>,
    need: Need,
    mode: Mode,
    number: String,
    events: Vec<Event>,
}

impl Default for Pda {
    fn default() -> Pda {
        Pda::new()
    }
}

impl Pda {
    pub fn new() -> Pda {
        Pda {
            stack: Vec::new(),
            need: Need::Value,
            mode: Mode::Structure,
            number: String::new(),
            events: Vec::new(),
        }
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn last_event(&self) -> Option<Event> {
        self.events.last().copied()
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// `false` once the machine has seen something that can never become JSON.
    pub fn is_alive(&self) -> bool {
        self.mode != Mode::Dead
    }

    /// A finished, complete document (no pending number, no open container).
    pub fn is_complete(&self) -> bool {
        self.mode == Mode::Structure && self.need == Need::Stop
    }

    pub fn push_str(&mut self, text: &str) -> bool {
        for ch in text.chars() {
            if !self.step(ch) {
                return false;
            }
        }
        true
    }

    pub fn step(&mut self, ch: char) -> bool {
        match self.mode.clone() {
            Mode::String { role } => self.step_string(ch, role),
            Mode::Escape { role } => self.step_escape(ch, role),
            Mode::Unicode { role, left } => self.step_unicode(ch, role, left),
            Mode::Number => self.step_number(ch),
            Mode::Literal { word, matched } => self.step_literal(ch, word, matched),
            Mode::Structure => self.step_structure(ch),
            Mode::Dead => false,
        }
    }

    fn die(&mut self) -> bool {
        self.mode = Mode::Dead;
        false
    }

    fn step_structure(&mut self, ch: char) -> bool {
        if ch.is_ascii_whitespace() {
            return true;
        }
        match self.need {
            Need::Stop => self.die(),
            Need::Value | Need::ValueOrClose => {
                if self.need == Need::ValueOrClose {
                    match (ch, self.stack.last()) {
                        (']', Some(Frame::Array)) => return self.close_container(),
                        ('}', Some(Frame::Object)) => return self.close_container(),
                        _ => {}
                    }
                }
                self.start_value(ch)
            }
            Need::Key | Need::KeyOrClose => {
                if self.need == Need::KeyOrClose && ch == '}' {
                    return self.close_container();
                }
                if ch == '"' {
                    self.mode = Mode::String { role: Role::Key };
                    return true;
                }
                self.die()
            }
            Need::Colon => {
                if ch == ':' {
                    self.need = Need::Value;
                    true
                } else {
                    self.die()
                }
            }
            Need::Comma => match (ch, self.stack.last().copied()) {
                (']', Some(Frame::Array)) | ('}', Some(Frame::Object)) => self.close_container(),
                (',', Some(Frame::Array)) => {
                    self.need = Need::Value;
                    self.events.push(Event::Separator);
                    true
                }
                (',', Some(Frame::Object)) => {
                    self.need = Need::Key;
                    self.events.push(Event::Separator);
                    true
                }
                _ => self.die(),
            },
        }
    }

    fn start_value(&mut self, ch: char) -> bool {
        match ch {
            '"' => {
                self.mode = Mode::String { role: Role::Value };
                true
            }
            't' => {
                self.mode = Mode::Literal {
                    word: "true",
                    matched: 1,
                };
                true
            }
            'f' => {
                self.mode = Mode::Literal {
                    word: "false",
                    matched: 1,
                };
                true
            }
            'n' => {
                self.mode = Mode::Literal {
                    word: "null",
                    matched: 1,
                };
                true
            }
            '[' => {
                self.stack.push(Frame::Array);
                self.need = Need::ValueOrClose;
                self.events.push(Event::Open);
                true
            }
            '{' => {
                self.stack.push(Frame::Object);
                self.need = Need::KeyOrClose;
                self.events.push(Event::Open);
                true
            }
            '-' | '0'..='9' => {
                self.mode = Mode::Number;
                self.number.clear();
                self.number.push(ch);
                true
            }
            _ => self.die(),
        }
    }

    fn close_container(&mut self) -> bool {
        self.stack.pop();
        self.events.push(Event::Close);
        self.complete_value()
    }

    /// A value just ended (or a container closed); figure out what may follow.
    fn complete_value(&mut self) -> bool {
        self.mode = Mode::Structure;
        self.need = if self.stack.is_empty() {
            Need::Stop
        } else {
            Need::Comma
        };
        self.events.push(Event::Value);
        true
    }

    fn step_string(&mut self, ch: char, role: Role) -> bool {
        match ch {
            '"' => match role {
                Role::Key => {
                    self.mode = Mode::Structure;
                    self.need = Need::Colon;
                    self.events.push(Event::Key);
                    true
                }
                Role::Value => self.complete_value(),
            },
            '\\' => {
                self.mode = Mode::Escape { role };
                true
            }
            ch if (ch as u32) < 0x20 => self.die(),
            _ => true,
        }
    }

    fn step_escape(&mut self, ch: char, role: Role) -> bool {
        match ch {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                self.mode = Mode::String { role };
                true
            }
            'u' => {
                self.mode = Mode::Unicode { role, left: 4 };
                true
            }
            _ => self.die(),
        }
    }

    fn step_unicode(&mut self, ch: char, role: Role, left: u8) -> bool {
        if !ch.is_ascii_hexdigit() {
            return self.die();
        }
        if left <= 1 {
            self.mode = Mode::String { role };
        } else {
            self.mode = Mode::Unicode { role, left: left - 1 };
        }
        true
    }

    fn step_number(&mut self, ch: char) -> bool {
        if ch.is_ascii_digit() || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E' {
            self.number.push(ch);
            if !number_prefix_ok(&self.number, false) {
                return self.die();
            }
            return true;
        }
        // The number ends here: it must be complete, then this char is structural again.
        if !number_prefix_ok(&self.number, true) {
            return self.die();
        }
        self.number.clear();
        self.complete_value();
        self.step(ch)
    }

    fn step_literal(&mut self, ch: char, word: &'static str, matched: usize) -> bool {
        let bytes = word.as_bytes();
        if matched >= bytes.len() || ch != bytes[matched] as char {
            return self.die();
        }
        if matched + 1 == bytes.len() {
            self.complete_value()
        } else {
            self.mode = Mode::Literal {
                word,
                matched: matched + 1,
            };
            true
        }
    }

    /// Close a pending number and require a complete document; `None` means `self` cannot
    /// become one.
    pub fn finish(&self) -> Option<Pda> {
        let mut finished = self.clone();
        if finished.mode == Mode::Number {
            if !number_prefix_ok(&finished.number, true) {
                return None;
            }
            finished.number.clear();
            finished.complete_value();
        }
        if finished.is_complete() {
            Some(finished)
        } else {
            None
        }
    }

    pub fn accepts(&self) -> bool {
        self.finish().is_some()
    }

    /// Indices of tokens that keep the document alive.
    pub fn candidates(&self, vocabulary: &[String]) -> Vec<u32> {
        let mut allowed = Vec::new();
        for (index, token) in vocabulary.iter().enumerate() {
            let mut probe = self.clone();
            if probe.push_str(token) {
                allowed.push(index as u32);
            }
        }
        allowed
    }

    /// Candidates plus the end-of-sequence token, which is only legal on a finished document.
    pub fn mask(&self, vocabulary: &[String], eos: Option<&str>) -> Vec<u32> {
        let mut allowed = self.candidates(vocabulary);
        let finished = self.finish().is_some();
        if finished {
            for (index, token) in vocabulary.iter().enumerate() {
                if Some(token.as_str()) == eos {
                    allowed.push(index as u32);
                }
            }
        }
        allowed.sort_unstable();
        allowed.dedup();
        allowed
    }
}

/// Is `text` a prefix of a JSON number (`complete = false`) or a whole one (`complete = true`)?
pub fn number_prefix_ok(text: &str, complete: bool) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut at = 0usize;
    if bytes[at] == b'-' {
        at += 1;
        if at == bytes.len() {
            return !complete;
        }
    }
    match bytes[at] {
        b'0' => at += 1,
        b'1'..=b'9' => {
            while at < bytes.len() && bytes[at].is_ascii_digit() {
                at += 1;
            }
        }
        _ => return false,
    }
    if at == bytes.len() {
        return true;
    }
    if bytes[at] == b'.' {
        at += 1;
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == bytes.len() {
            return !complete || at > start;
        }
        if at == start {
            return false;
        }
    }
    if at < bytes.len() && (bytes[at] == b'e' || bytes[at] == b'E') {
        at += 1;
        if at < bytes.len() && (bytes[at] == b'+' || bytes[at] == b'-') {
            at += 1;
        }
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == bytes.len() {
            return !complete || at > start;
        }
        if at == start {
            return false;
        }
    }
    at == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::prefix_status;
    use crate::json::Prefix;

    fn feed(text: &str) -> Pda {
        let mut pda = Pda::new();
        pda.push_str(text);
        pda
    }

    #[test]
    fn accepts_complete_documents_only() {
        for text in [
            "1",
            "-0.5",
            "1e3",
            "\"x\"",
            "true",
            "null",
            "[]",
            "{}",
            "{\"a\": [1, 2, {\"b\": null}], \"c\": false}",
            "  { \"a\" : 1 }  ",
        ] {
            let pda = feed(text);
            assert!(pda.accepts(), "should accept {:?}", text);
        }
        for text in ["", "{", "{\"a\"", "{\"a\":1,}", "[1,", "\"abc", "1e", "-", "truex"] {
            let pda = feed(text);
            assert!(!pda.accepts(), "should not accept {:?}", text);
        }
    }

    #[test]
    fn events_track_structure() {
        let pda = feed("{\"a\": [1]}");
        assert_eq!(pda.last_event(), Some(Event::Value));
        assert!(pda.events().contains(&Event::Key));
        assert_eq!(
            pda.events().iter().filter(|event| **event == Event::Close).count(),
            1
        );
        assert_eq!(pda.depth(), 0);
    }

    #[test]
    fn prefix_verdicts_match_the_json_recognizer() {
        let samples = [
            "",
            " ",
            "{",
            "{}",
            "{\"a\"",
            "{\"a\":",
            "{\"a\": ",
            "{\"a\": 1",
            "{\"a\": 1}",
            "{\"a\" 1}",
            "[1,]",
            "[1",
            "nul",
            "null",
            "nullx",
            "\"a\\",
            "\"a\\u00",
            "0.",
            "01",
            "-",
            "1e",
            "1e+",
            "1e+2",
            "abc",
        ];
        for sample in samples {
            let pda = feed(sample);
            let expected = match prefix_status(sample) {
                Prefix::Complete => Prefix::Complete,
                Prefix::Incomplete => Prefix::Incomplete,
                Prefix::Invalid => Prefix::Invalid,
            };
            let actual = if pda.accepts() {
                Prefix::Complete
            } else if pda.is_alive() {
                Prefix::Incomplete
            } else {
                Prefix::Invalid
            };
            assert_eq!(actual, expected, "diverged on {:?}", sample);
        }
    }

    #[test]
    fn mask_admits_only_parseable_tokens() {
        let vocabulary: Vec<String> = [
            "{\"a\": ",
            "}",
            "\"a\"",
            ": ",
            "1",
            "]",
            "null",
            "<stop>",
        ]
        .iter()
        .map(|token| token.to_string())
        .collect();
        let empty = Pda::new();
        let allowed = empty.mask(&vocabulary, Some("<stop>"));
        assert!(allowed.contains(&0));
        assert!(!allowed.contains(&6));
        assert!(!allowed.contains(&7));

        let open = feed("{\"a\": ");
        let allowed = open.mask(&vocabulary, Some("<stop>"));
        assert!(allowed.contains(&4));
        assert!(allowed.contains(&6));
        assert!(!allowed.contains(&1));

        let done = feed("{\"a\": 1}");
        assert_eq!(done.mask(&vocabulary, Some("<stop>")), vec![7]);
    }

    #[test]
    fn number_prefixes_follow_rfc_8259() {
        for text in ["0", "-0", "1", "-1.5", "1e3", "1E+3", "0.25"] {
            assert!(number_prefix_ok(text, true), "{} should be a number", text);
        }
        for text in ["", "-", "01", "1.", ".5", "1e", "1e+", "+1", "1x", "00"] {
            assert!(!number_prefix_ok(text, true), "{} is not a number", text);
        }
        for text in ["", "-", "1.", "1e", "1e+", "0."] {
            assert!(number_prefix_ok(text, false), "{} is a legal prefix", text);
        }
    }

    #[test]
    fn literals_are_consumed_incrementally() {
        let mut pda = Pda::new();
        assert!(pda.push_str("tru"));
        assert!(pda.is_alive() && !pda.accepts());
        assert!(pda.push_str("e"));
        assert!(pda.accepts());
        let mut broken = Pda::new();
        assert!(!broken.push_str("trux"));
        assert!(!broken.is_alive());
    }
}
