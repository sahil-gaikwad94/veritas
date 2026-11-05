//! SMT-LIB2 bridge: emit the script, or hand it to an external solver.
//!
//! Port of `python/vexx/smt_bridge.py`. Nothing here links against Z3: the crate stays
//! dependency-free, so the bridge is
//!
//! * a **script emitter** — the linear fragment this engine decides, written as SMT-LIB2 with
//!   named assertions, so any solver (Z3, cvc5, …) can check or cross-check it;
//! * a **runtime delegation** — when a solver binary is configured *and* present, the script is
//!   piped to it and the answer is parsed.
//!
//! Without a binary the status is [`SmtStatus::Unavailable`], never a guess. That is why the
//! Python engine can ship this module by default: the fast path is native, and the SMT path is
//! strictly additive.

use crate::expr::CmpOp;
use crate::solver::Linear;
use std::collections::BTreeSet;
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmtStatus {
    Sat,
    Unsat,
    Unknown,
    /// No solver binary was configured or runnable.
    Unavailable,
}

impl SmtStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SmtStatus::Sat => "sat",
            SmtStatus::Unsat => "unsat",
            SmtStatus::Unknown => "unknown",
            SmtStatus::Unavailable => "unavailable",
        }
    }

    fn parse(text: &str) -> SmtStatus {
        match text.trim() {
            "sat" => SmtStatus::Sat,
            "unsat" => SmtStatus::Unsat,
            _ => SmtStatus::Unknown,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmtResult {
    pub status: Option<SmtStatus>,
    pub core: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SmtBridge {
    binary: Option<String>,
    logic: String,
    timeout_ms: u64,
    declarations: BTreeSet<u32>,
    assertions: Vec<(String, String)>,
}

impl Default for SmtBridge {
    fn default() -> SmtBridge {
        SmtBridge::new(None)
    }
}

impl SmtBridge {
    pub fn new(binary: Option<&str>) -> SmtBridge {
        SmtBridge {
            binary: binary.map(|path| path.to_string()),
            logic: "QF_LRA".to_string(),
            timeout_ms: 5000,
            declarations: BTreeSet::new(),
            assertions: Vec::new(),
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> SmtBridge {
        self.timeout_ms = timeout_ms;
        self
    }

    pub fn binary(&self) -> Option<&str> {
        self.binary.as_deref()
    }

    /// True when the bridge can actually run a solver.
    pub fn available(&self) -> bool {
        match &self.binary {
            Some(path) => Command::new(path)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false),
            None => false,
        }
    }

    fn declare(&mut self, var: u32) {
        self.declarations.insert(var);
    }

    /// Add `Σ coeffs * x <op> rhs` with an optional assertion name.
    pub fn assert_linear(&mut self, constraint: &Linear, name: Option<&str>) {
        for (var, _) in &constraint.coeffs {
            self.declare(*var);
        }
        let mut body = String::from("(+");
        for (var, coeff) in &constraint.coeffs {
            body.push_str(&format!(" (* {} x{})", render_real(*coeff), var));
        }
        body.push(')');
        if constraint.coeffs.is_empty() {
            body = "0.0".to_string();
        }
        let operator = match constraint.op {
            CmpOp::Eq => "=",
            CmpOp::Le => "<=",
            CmpOp::Lt => "<",
            CmpOp::Ge => ">=",
            CmpOp::Gt => ">",
            CmpOp::Ne => "distinct",
            CmpOp::In | CmpOp::NotIn => {
                // Not a linear comparison: leave it to the caller instead of emitting nonsense.
                return;
            }
        };
        let formula = if operator == "distinct" {
            format!("(not (= {} {}))", body, render_real(constraint.rhs))
        } else {
            format!("({} {} {})", operator, body, render_real(constraint.rhs))
        };
        let formula = match name {
            Some(label) => format!("(! {} :named {})", formula, sanitize(label)),
            None => formula,
        };
        self.assertions.push((formula, name.unwrap_or("").to_string()));
    }

    /// Tracked assertion: the name is what `(get-unsat-core)` reports back.
    pub fn assert_and_track(&mut self, constraint: &Linear, name: &str) {
        self.assert_linear(constraint, Some(name));
    }

    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }

    pub fn script(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("(set-logic {})\n", self.logic));
        out.push_str("(set-option :produce-unsat-cores true)\n");
        out.push_str(&format!("(set-option :timeout {})\n", self.timeout_ms));
        for var in &self.declarations {
            out.push_str(&format!("(declare-fun x{} () Real)\n", var));
        }
        for (formula, _) in &self.assertions {
            out.push_str(&format!("(assert {})\n", formula));
        }
        out.push_str("(check-sat)\n");
        out.push_str("(get-unsat-core)\n");
        out
    }

    /// Run the configured solver, or report why not.
    pub fn check(&self) -> SmtResult {
        let binary = match &self.binary {
            Some(path) => path.clone(),
            None => {
                return SmtResult {
                    status: Some(SmtStatus::Unavailable),
                    core: Vec::new(),
                    reason: Some("no SMT solver binary configured".to_string()),
                };
            }
        };
        let mut child = match Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                return SmtResult {
                    status: Some(SmtStatus::Unavailable),
                    core: Vec::new(),
                    reason: Some(format!("could not run {}: {}", binary, error)),
                };
            }
        };
        if let Some(stdin) = child.stdin.as_mut() {
            if stdin.write_all(self.script().as_bytes()).is_err() {
                return SmtResult {
                    status: Some(SmtStatus::Unknown),
                    core: Vec::new(),
                    reason: Some("could not write the SMT-LIB script".to_string()),
                };
            }
        }
        match child.wait_with_output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
                let status = lines
                    .next()
                    .map(SmtStatus::parse)
                    .unwrap_or(SmtStatus::Unknown);
                let core = lines
                    .next()
                    .map(parse_core)
                    .unwrap_or_default();
                SmtResult {
                    status: Some(status),
                    core,
                    reason: None,
                }
            }
            Err(error) => SmtResult {
                status: Some(SmtStatus::Unknown),
                core: Vec::new(),
                reason: Some(format!("solver failed: {}", error)),
            },
        }
    }
}

fn render_real(value: f64) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    if value == value.trunc() {
        format!("{}.0", value as i64)
    } else {
        format!("{:?}", value)
    }
}

fn sanitize(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' })
        .collect();
    if out.is_empty() || out.starts_with(|ch: char| ch.is_ascii_digit()) {
        out = format!("c_{}", out);
    }
    out
}

/// Parse `(c0 c2)` into `["c0", "c2"]`, keeping `|quoted symbols|` intact.
fn parse_core(line: &str) -> Vec<String> {
    let inner = line.trim().trim_start_matches('(').trim_end_matches(')');
    let mut names: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in inner.chars() {
        match ch {
            '|' => quoted = !quoted,
            ch if ch.is_whitespace() && !quoted => {
                if !current.is_empty() {
                    names.push(std::mem::take(&mut current));
                }
            }
            ch => current.push(ch),
        }
    }
    if !current.is_empty() {
        names.push(current);
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(coeffs: &[(u32, f64)], rhs: f64, op: CmpOp) -> Linear {
        Linear {
            coeffs: coeffs.to_vec(),
            rhs,
            op,
        }
    }

    #[test]
    fn scripts_declare_every_variable_and_name_every_assertion() {
        let mut bridge = SmtBridge::new(None);
        bridge.assert_and_track(&atom(&[(0, 1.0), (1, 1.0)], 1.0, CmpOp::Le), "weights_sum");
        bridge.assert_and_track(&atom(&[(0, 1.0)], 0.05, CmpOp::Ge), "w0_floor");
        let script = bridge.script();
        assert!(script.starts_with("(set-logic QF_LRA)\n"));
        assert!(script.contains("(declare-fun x0 () Real)"));
        assert!(script.contains("(declare-fun x1 () Real)"));
        assert!(script.contains("(! (<= (+ (* 1.0 x0) (* 1.0 x1)) 1.0) :named weights_sum)"));
        assert!(script.contains("(check-sat)"));
        assert!(script.contains("(get-unsat-core)"));
        assert_eq!(bridge.assertion_count(), 2);
    }

    #[test]
    fn non_linear_operators_are_skipped_not_mangled() {
        let mut bridge = SmtBridge::new(None);
        bridge.assert_linear(&atom(&[(0, 1.0)], 1.0, CmpOp::In), Some("membership"));
        assert_eq!(bridge.assertion_count(), 0);
        bridge.assert_linear(&atom(&[(0, 1.0)], 1.0, CmpOp::Eq), Some("pinned"));
        assert!(bridge.script().contains("(= (+ (* 1.0 x0)) 1.0)"));
    }

    #[test]
    fn without_a_binary_the_bridge_says_unavailable() {
        let bridge = SmtBridge::new(None);
        assert!(!bridge.available());
        let result = bridge.check();
        assert_eq!(result.status, Some(SmtStatus::Unavailable));
        assert!(result.reason.is_some());
    }

    #[test]
    fn missing_binaries_are_reported_not_panicked() {
        let bridge = SmtBridge::new(Some("/nonexistent/z3"));
        assert!(!bridge.available());
        assert_eq!(bridge.check().status, Some(SmtStatus::Unavailable));
    }

    #[test]
    fn cores_and_names_are_parsed() {
        assert_eq!(parse_core("(c0 c2)"), vec!["c0", "c2"]);
        assert_eq!(parse_core("(|a b|)"), vec!["a b"]);
        assert!(parse_core("()").is_empty());
        assert_eq!(sanitize("w0 floor"), "w0_floor");
        assert_eq!(sanitize("9lives"), "c_9lives");
    }
}
