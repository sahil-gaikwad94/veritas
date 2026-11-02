//! Deterministic 0-DoF evaluation, cascaded scheduling and Jump-Forward splicing.
//!
//! Port of `python/veritas/zero_dof.py`. A field is *0-DoF* when every input of its derivation
//! is already bound: nothing is left to decide, so the value is computed exactly, formatted,
//! and spliced into the token stream in one batched prefill.
//!
//! Values are numeric, because the native path exists to accelerate arithmetic derivations;
//! derivations that produce strings, booleans or schema-formatted text stay in Python. A rule's
//! [`Expr`] evaluates to the *value of the target*, and its variable ids index into the rule's
//! `inputs` (the Python side renumbers the wire expression accordingly).

use crate::expr::{Env, Expr};
use crate::json::{json_number, render_string};
use std::collections::BTreeMap;

/// Bound values, keyed by the JSON field name (the same key the splice emits).
pub type Values = BTreeMap<String, f64>;

/// One `target = expr(inputs)` derivation.
#[derive(Clone, Debug)]
pub struct ZeroDofRule {
    /// Field name written by the splice.
    pub target: String,
    /// Dependency names; `Expr::Var(i)` refers to `inputs[i]`.
    pub inputs: Vec<String>,
    pub expr: Expr,
    pub precision_digits: u32,
}

impl ZeroDofRule {
    pub fn new(target: &str, inputs: Vec<String>, expr: Expr, precision_digits: u32) -> ZeroDofRule {
        ZeroDofRule {
            target: target.to_string(),
            inputs,
            expr,
            precision_digits,
        }
    }

    fn env(&self, values: &Values) -> Option<Env> {
        let mut env: Env = Env::new();
        for (index, name) in self.inputs.iter().enumerate() {
            env.insert(index as u32, *values.get(name)?);
        }
        Some(env)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ZeroDofEngine {
    rules: Vec<ZeroDofRule>,
    last_splice: String,
    jump_forward_fields: Vec<String>,
}

impl ZeroDofEngine {
    pub fn new(rules: Vec<ZeroDofRule>) -> ZeroDofEngine {
        ZeroDofEngine {
            rules,
            last_splice: String::new(),
            jump_forward_fields: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: ZeroDofRule) {
        self.rules.push(rule);
    }

    pub fn rules(&self) -> &[ZeroDofRule] {
        &self.rules
    }

    pub fn last_splice(&self) -> &str {
        &self.last_splice
    }

    pub fn jump_forward_fields(&self) -> &[String] {
        &self.jump_forward_fields
    }

    fn rule_for(&self, target: &str) -> Option<&ZeroDofRule> {
        self.rules.iter().find(|rule| rule.target == target)
    }

    /// True when `target` has a rule and every dependency is already bound.
    pub fn is_ready(&self, target: &str, values: &Values) -> bool {
        match self.rule_for(target) {
            Some(rule) => rule.inputs.iter().all(|name| values.contains_key(name)),
            None => false,
        }
    }

    /// Rules whose target is unbound but whose dependencies are all bound.
    pub fn pending(&self, values: &Values) -> Vec<&ZeroDofRule> {
        self.rules
            .iter()
            .filter(|rule| !values.contains_key(&rule.target))
            .filter(|rule| rule.inputs.iter().all(|name| values.contains_key(name)))
            .collect()
    }

    /// Evaluate one target; `None` means "not determinable from these values".
    pub fn evaluate(&self, target: &str, values: &Values) -> Option<f64> {
        let rule = self.rule_for(target)?;
        let env = rule.env(values)?;
        rule.expr.eval_number(&env)
    }

    /// Fixpoint over all rules: evaluate everything that becomes ready.
    pub fn evaluate_cascade(&self, values: &Values) -> Values {
        let mut result = values.clone();
        for _ in 0..100 {
            let mut changed = false;
            for rule in &self.rules {
                if result.contains_key(&rule.target) {
                    continue;
                }
                let env = match rule.env(&result) {
                    Some(env) => env,
                    None => continue,
                };
                if let Some(value) = rule.expr.eval_number(&env) {
                    result.insert(rule.target.clone(), value);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        result
    }

    /// Evaluate every currently-ready rule once; returns the newly computed fields in the
    /// order the rules became ready.
    pub fn compute_ready(&self, values: &Values) -> Vec<(String, f64)> {
        let mut computed: Vec<(String, f64)> = Vec::new();
        let mut result = values.clone();
        for _ in 0..100 {
            let mut progress = false;
            let ready: Vec<String> = self
                .pending(&result)
                .into_iter()
                .map(|rule| rule.target.clone())
                .collect();
            for target in ready {
                let rule = match self.rule_for(&target) {
                    Some(rule) => rule.clone(),
                    None => continue,
                };
                let env = match rule.env(&result) {
                    Some(env) => env,
                    None => continue,
                };
                if let Some(value) = rule.expr.eval_number(&env) {
                    let value = round_to(value, rule.precision_digits);
                    computed.push((target.clone(), value));
                    result.insert(target, value);
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
        computed
    }

    /// The Jump-Forward continuation text plus the fields it computed.
    ///
    /// Mirrors `render_continuation`: `, "k": v, "k2": v2` — the leading comma continues the
    /// object the model has already opened.
    pub fn splice_json(
        &mut self,
        values: &Values,
        keys: Option<&[String]>,
        precision: Option<&BTreeMap<String, u32>>,
    ) -> (String, Vec<(String, f64)>) {
        let computed = self.compute_ready(values);
        self.jump_forward_fields = computed.iter().map(|(key, _)| key.clone()).collect();
        if computed.is_empty() {
            self.last_splice = String::new();
            return (String::new(), computed);
        }
        let mut precision_map: BTreeMap<String, u32> = precision.cloned().unwrap_or_default();
        for rule in &self.rules {
            precision_map
                .entry(rule.target.clone())
                .or_insert(rule.precision_digits);
        }
        let mut ordered: Vec<String> = Vec::new();
        if let Some(keys) = keys {
            for key in keys {
                if computed.iter().any(|(name, _)| name == key) && !ordered.contains(key) {
                    ordered.push(key.clone());
                }
            }
        }
        if ordered.is_empty() {
            ordered = computed.iter().map(|(key, _)| key.clone()).collect();
        }
        let mut text = String::new();
        for key in &ordered {
            if let Some((_, value)) = computed.iter().find(|(name, _)| name == key) {
                let digits = precision_map.get(key).copied();
                text.push_str(&format!(
                    ", {}: {}",
                    render_string(key),
                    json_number(*value, digits)
                ));
            }
        }
        self.last_splice = text.clone();
        (text, computed)
    }
}

fn round_to(value: f64, digits: u32) -> f64 {
    if digits == 0 {
        return value;
    }
    crate::json::round_to(value, digits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn rule(target: &str, inputs: &[&str], expr: &str, precision: u32) -> ZeroDofRule {
        ZeroDofRule::new(
            target,
            inputs.iter().map(|name| name.to_string()).collect(),
            Expr::from_json(&parse(expr).expect("expression json")),
            precision,
        )
    }

    fn values(pairs: &[(&str, f64)]) -> Values {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), *value))
            .collect()
    }

    // sharpe = (expected_return - rf) / volatility, with Var(0) = expected_return etc.
    fn sharpe_rule() -> ZeroDofRule {
        rule(
            "sharpe_ratio",
            &["expected_return", "rf", "volatility"],
            "{\"t\":\"div\",\"a\":{\"t\":\"sub\",\"a\":{\"t\":\"var\",\"id\":0},\"b\":{\"t\":\"var\",\"id\":1}},\"b\":{\"t\":\"var\",\"id\":2}}",
            5,
        )
    }

    #[test]
    fn ready_rules_cascade_in_topological_order() {
        let engine = ZeroDofEngine::new(vec![
            rule(
                "mean",
                &["a", "b"],
                "{\"t\":\"div\",\"a\":{\"t\":\"sum\",\"a\":[{\"t\":\"var\",\"id\":0},{\"t\":\"var\",\"id\":1}]},\"b\":{\"t\":\"const\",\"v\":2.0}}",
                4,
            ),
            rule(
                "spread",
                &["mean", "a"],
                "{\"t\":\"sub\",\"a\":{\"t\":\"var\",\"id\":1},\"b\":{\"t\":\"var\",\"id\":0}}",
                4,
            ),
            sharpe_rule(),
        ]);
        assert!(!engine.is_ready("spread", &values(&[("a", 1.0), ("b", 3.0)])));
        assert!(engine.is_ready("mean", &values(&[("a", 1.0), ("b", 3.0)])));
        let computed = engine.compute_ready(&values(&[("a", 1.0), ("b", 3.0)]));
        let names: Vec<&str> = computed.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(names, vec!["mean", "spread"]);
        assert_eq!(computed[0].1, 2.0);
        assert_eq!(computed[1].1, 1.0);
        assert_eq!(engine.jump_forward_fields(), ["mean", "spread"]);
    }

    #[test]
    fn splicing_renders_a_json_continuation() {
        let mut engine = ZeroDofEngine::new(vec![sharpe_rule()]);
        let bound = values(&[("expected_return", 0.08), ("rf", 0.02), ("volatility", 0.09)]);
        let (text, computed) = engine.splice_json(&bound, None, None);
        assert_eq!(computed.len(), 1);
        // (0.08 - 0.02) / 0.09 = 0.66666...
        assert_eq!(text, ", \"sharpe_ratio\": 0.66667");
        assert_eq!(engine.last_splice(), text);
    }

    #[test]
    fn missing_inputs_produce_no_splice() {
        let mut engine = ZeroDofEngine::new(vec![sharpe_rule()]);
        let (text, computed) = engine.splice_json(&values(&[("expected_return", 0.08)]), None, None);
        assert!(text.is_empty() && computed.is_empty());
        assert!(engine.last_splice().is_empty());
    }

    #[test]
    fn key_order_and_precision_are_respected() {
        let mut engine = ZeroDofEngine::new(vec![
            rule("one", &["x"], "{\"t\":\"const\",\"v\":1.0}", 2),
            rule("two", &["x"], "{\"t\":\"add\",\"a\":{\"t\":\"var\",\"id\":0},\"b\":{\"t\":\"const\",\"v\":0.5}}", 3),
        ]);
        let mut precision = BTreeMap::new();
        precision.insert("one".to_string(), 3u32);
        let keys = vec!["two".to_string(), "one".to_string()];
        let bound = values(&[("x", 2.0)]);
        let (text, _) = engine.splice_json(&bound, Some(&keys), Some(&precision));
        assert_eq!(text, ", \"two\": 2.500, \"one\": 1.000");
    }

    #[test]
    fn a_dependency_cycle_terminates_without_a_value() {
        let mut engine = ZeroDofEngine::new(vec![
            rule("a", &["b"], "{\"t\":\"var\",\"id\":0}", 3),
            rule("b", &["a"], "{\"t\":\"var\",\"id\":0}", 3),
        ]);
        let (text, computed) = engine.splice_json(&values(&[]), None, None);
        assert!(text.is_empty() && computed.is_empty());
    }
}
