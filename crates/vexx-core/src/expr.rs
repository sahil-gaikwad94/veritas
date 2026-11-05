//! Tri-state expression evaluation over a JSON wire format.
//!
//! The Python compiler lowers a schema to VIR; `python/vexx/native.py` re-encodes the
//! parts the native core can decide into the compact form parsed here. Anything outside the
//! whitelist parses to [`Expr::Unsupported`] and evaluates to [`Tri::Unknown`] — never to a
//! silent `true`, because the concrete verifier stays authoritative.

use crate::json::Json;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    True,
    False,
    Unknown,
}

#[allow(clippy::should_implement_trait)] // Kleene negation, also available via `!`
impl Tri {
    pub fn and(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::False, _) | (_, Tri::False) => Tri::False,
            (Tri::True, Tri::True) => Tri::True,
            _ => Tri::Unknown,
        }
    }

    pub fn or(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::True, _) | (_, Tri::True) => Tri::True,
            (Tri::False, Tri::False) => Tri::False,
            _ => Tri::Unknown,
        }
    }

    pub fn not(self) -> Tri {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }

    pub fn as_bool(self) -> Option<bool> {
        match self {
            Tri::True => Some(true),
            Tri::False => Some(false),
            Tri::Unknown => None,
        }
    }
}

impl std::ops::Not for Tri {
    type Output = Tri;

    fn not(self) -> Tri {
        Tri::not(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

impl CmpOp {
    pub fn parse(text: &str) -> Option<CmpOp> {
        match text {
            "eq" | "==" => Some(CmpOp::Eq),
            "ne" | "!=" => Some(CmpOp::Ne),
            "lt" | "<" => Some(CmpOp::Lt),
            "le" | "<=" => Some(CmpOp::Le),
            "gt" | ">" => Some(CmpOp::Gt),
            "ge" | ">=" => Some(CmpOp::Ge),
            "in" => Some(CmpOp::In),
            "not_in" => Some(CmpOp::NotIn),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::In => "in",
            CmpOp::NotIn => "not in",
        }
    }

    /// Negation used by the contractor and by preflight refutation.
    pub fn negate(self) -> CmpOp {
        match self {
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::In => CmpOp::NotIn,
            CmpOp::NotIn => CmpOp::In,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Const(f64),
    Var(u32),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Sum(Vec<Expr>),
    Call(String, Vec<Expr>),
    Compare {
        op: CmpOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        tol: f64,
    },
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
    IfElse(Box<Expr>, Box<Expr>, Box<Expr>),
    Unsupported(String),
}

pub type Env = HashMap<u32, f64>;

impl Expr {
    pub fn from_json(node: &Json) -> Expr {
        let node = match node {
            Json::Object(_) => node,
            other => return Expr::Unsupported(format!("expression must be an object: {}", other.render())),
        };
        let tag = match node.get("t").and_then(|value| value.as_str()) {
            Some(tag) => tag.to_string(),
            None => return Expr::Unsupported("expression has no 't' tag".to_string()),
        };
        match tag.as_str() {
            "const" => match node.get("v") {
                Some(Json::Number(value)) => Expr::Const(*value),
                Some(Json::Bool(value)) => Expr::Const(if *value { 1.0 } else { 0.0 }),
                _ => Expr::Unsupported("const without a numeric 'v'".to_string()),
            },
            "var" => match node.get("id") {
                Some(Json::Number(value)) => Expr::Var(*value as u32),
                _ => Expr::Unsupported("var without a numeric 'id'".to_string()),
            },
            "add" | "sub" | "mul" | "div" | "pow" | "min" | "max" => {
                let left = match node.get("a") {
                    Some(value) => Expr::from_json(value),
                    None => return Expr::Unsupported(format!("{} without 'a'", tag)),
                };
                let right = match node.get("b") {
                    Some(value) => Expr::from_json(value),
                    None => return Expr::Unsupported(format!("{} without 'b'", tag)),
                };
                let left = Box::new(left);
                let right = Box::new(right);
                match tag.as_str() {
                    "add" => Expr::Add(left, right),
                    "sub" => Expr::Sub(left, right),
                    "mul" => Expr::Mul(left, right),
                    "div" => Expr::Div(left, right),
                    "pow" => Expr::Pow(left, right),
                    "min" => Expr::Min(left, right),
                    _ => Expr::Max(left, right),
                }
            }
            "neg" | "sqrt" | "abs" | "not" => {
                let inner = match node.get("a") {
                    Some(value) => Box::new(Expr::from_json(value)),
                    None => return Expr::Unsupported(format!("{} without 'a'", tag)),
                };
                match tag.as_str() {
                    "neg" => Expr::Neg(inner),
                    "sqrt" => Expr::Sqrt(inner),
                    "abs" => Expr::Abs(inner),
                    _ => Expr::Not(inner),
                }
            }
            "sum" | "and" | "or" => {
                let args = match node.get("a") {
                    Some(Json::Array(items)) => items.iter().map(Expr::from_json).collect(),
                    _ => return Expr::Unsupported(format!("{} without an array 'a'", tag)),
                };
                match tag.as_str() {
                    "sum" => Expr::Sum(args),
                    "and" => Expr::And(args),
                    _ => Expr::Or(args),
                }
            }
            "cmp" => {
                let op = match node
                    .get("op")
                    .and_then(|value| value.as_str())
                    .and_then(CmpOp::parse)
                {
                    Some(op) => op,
                    None => return Expr::Unsupported("cmp without a known 'op'".to_string()),
                };
                let lhs = match node.get("a") {
                    Some(value) => Box::new(Expr::from_json(value)),
                    None => return Expr::Unsupported("cmp without 'a'".to_string()),
                };
                let rhs = match node.get("b") {
                    Some(value) => Box::new(Expr::from_json(value)),
                    None => return Expr::Unsupported("cmp without 'b'".to_string()),
                };
                let tol = node
                    .get("tol")
                    .and_then(|value| value.as_f64())
                    .unwrap_or(0.0);
                Expr::Compare { op, lhs, rhs, tol }
            }
            "ite" => {
                let (test, then_value, else_value) = match (node.get("c"), node.get("a"), node.get("b")) {
                    (Some(test), Some(then_value), Some(else_value)) => (
                        Box::new(Expr::from_json(test)),
                        Box::new(Expr::from_json(then_value)),
                        Box::new(Expr::from_json(else_value)),
                    ),
                    _ => return Expr::Unsupported("ite without c/a/b".to_string()),
                };
                Expr::IfElse(test, then_value, else_value)
            }
            "call" => {
                let name = node
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = match node.get("a") {
                    Some(Json::Array(items)) => items.iter().map(Expr::from_json).collect(),
                    _ => Vec::new(),
                };
                Expr::Call(name, args)
            }
            other => Expr::Unsupported(format!("unsupported expression tag '{}'", other)),
        }
    }

    /// Numeric evaluation. `None` means "cannot decide", never zero.
    pub fn eval_number(&self, env: &Env) -> Option<f64> {
        let value = match self {
            Expr::Const(value) => *value,
            Expr::Var(id) => *env.get(id)?,
            Expr::Add(left, right) => left.eval_number(env)? + right.eval_number(env)?,
            Expr::Sub(left, right) => left.eval_number(env)? - right.eval_number(env)?,
            Expr::Mul(left, right) => left.eval_number(env)? * right.eval_number(env)?,
            Expr::Div(left, right) => {
                let divisor = right.eval_number(env)?;
                if divisor == 0.0 {
                    return None;
                }
                left.eval_number(env)? / divisor
            }
            Expr::Neg(inner) => -inner.eval_number(env)?,
            Expr::Pow(base, exponent) => {
                let base = base.eval_number(env)?;
                let exponent = exponent.eval_number(env)?;
                let result = base.powf(exponent);
                if result.is_nan() {
                    return None;
                }
                result
            }
            Expr::Sqrt(inner) => {
                let value = inner.eval_number(env)?;
                if value < 0.0 {
                    return None;
                }
                value.sqrt()
            }
            Expr::Abs(inner) => inner.eval_number(env)?.abs(),
            Expr::Min(left, right) => left.eval_number(env)?.min(right.eval_number(env)?),
            Expr::Max(left, right) => left.eval_number(env)?.max(right.eval_number(env)?),
            Expr::Sum(args) => {
                let mut total = 0.0;
                for arg in args {
                    total += arg.eval_number(env)?;
                }
                total
            }
            Expr::Call(name, args) => eval_call(name, args, env)?,
            Expr::Compare { .. } | Expr::And(_) | Expr::Or(_) | Expr::Not(_) => return None,
            Expr::IfElse(test, then_value, else_value) => {
                match test.eval(env).as_bool()? {
                    true => then_value.eval_number(env)?,
                    false => else_value.eval_number(env)?,
                }
            }
            Expr::Unsupported(_) => return None,
        };
        if value.is_finite() {
            Some(value)
        } else {
            None
        }
    }

    pub fn eval(&self, env: &Env) -> Tri {
        match self {
            Expr::Compare { op, lhs, rhs, tol } => compare(*op, lhs, rhs, *tol, env),
            Expr::And(args) => {
                let mut result = Tri::True;
                for arg in args {
                    result = result.and(arg.eval(env));
                }
                if args.is_empty() {
                    Tri::Unknown
                } else {
                    result
                }
            }
            Expr::Or(args) => {
                let mut result = Tri::False;
                for arg in args {
                    result = result.or(arg.eval(env));
                }
                if args.is_empty() {
                    Tri::Unknown
                } else {
                    result
                }
            }
            Expr::Not(inner) => inner.eval(env).not(),
            Expr::IfElse(test, then_value, else_value) => match test.eval(env) {
                Tri::True => then_value.eval(env),
                Tri::False => else_value.eval(env),
                Tri::Unknown => Tri::Unknown,
            },
            Expr::Unsupported(_) => Tri::Unknown,
            other => match other.eval_number(env) {
                Some(value) => {
                    if value != 0.0 {
                        Tri::True
                    } else {
                        Tri::False
                    }
                }
                None => Tri::Unknown,
            },
        }
    }

    /// Collect the variable ids referenced by this expression (depth first, in order).
    pub fn collect_vars(&self, out: &mut Vec<u32>) {
        match self {
            Expr::Var(id) => {
                if !out.contains(id) {
                    out.push(*id);
                }
            }
            Expr::Const(_) | Expr::Unsupported(_) => {}
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Neg(a) | Expr::Sqrt(a) | Expr::Abs(a) | Expr::Not(a) => a.collect_vars(out),
            Expr::Compare { lhs, rhs, .. } => {
                lhs.collect_vars(out);
                rhs.collect_vars(out);
            }
            Expr::IfElse(c, a, b) => {
                c.collect_vars(out);
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Sum(args) | Expr::And(args) | Expr::Or(args) | Expr::Call(_, args) => {
                for arg in args {
                    arg.collect_vars(out);
                }
            }
        }
    }
}

fn eval_call(name: &str, args: &[Expr], env: &Env) -> Option<f64> {
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        values.push(arg.eval_number(env)?);
    }
    match (name, values.as_slice()) {
        ("sqrt", [value]) => {
            if *value < 0.0 {
                None
            } else {
                Some(value.sqrt())
            }
        }
        ("abs", [value]) => Some(value.abs()),
        ("exp", [value]) => Some(value.exp()),
        ("log", [value]) => {
            if *value <= 0.0 {
                None
            } else {
                Some(value.ln())
            }
        }
        ("floor", [value]) => Some(value.floor()),
        ("ceil", [value]) => Some(value.ceil()),
        ("min", [left, right]) => Some(left.min(*right)),
        ("max", [left, right]) => Some(left.max(*right)),
        _ => None,
    }
}

fn compare(op: CmpOp, lhs: &Expr, rhs: &Expr, tol: f64, env: &Env) -> Tri {
    match op {
        CmpOp::In | CmpOp::NotIn => Tri::Unknown,
        _ => {
            let left = match lhs.eval_number(env) {
                Some(value) => value,
                None => return Tri::Unknown,
            };
            let right = match rhs.eval_number(env) {
                Some(value) => value,
                None => return Tri::Unknown,
            };
            let residual = left - right;
            let satisfied = match op {
                CmpOp::Eq => residual.abs() <= tol,
                CmpOp::Ne => residual.abs() > tol,
                CmpOp::Le => left <= right + tol,
                CmpOp::Lt => left < right + tol,
                CmpOp::Ge => left >= right - tol,
                CmpOp::Gt => left > right - tol,
                CmpOp::In | CmpOp::NotIn => return Tri::Unknown,
            };
            if satisfied {
                Tri::True
            } else {
                Tri::False
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn expr(source: &str) -> Expr {
        Expr::from_json(&parse(source).expect("valid expression json"))
    }

    #[test]
    fn evaluates_arithmetic() {
        let source = "{\"t\":\"cmp\",\"op\":\"le\",\"a\":{\"t\":\"add\",\"a\":{\"t\":\"var\",\"id\":0},\"b\":{\"t\":\"var\",\"id\":1}},\"b\":{\"t\":\"const\",\"v\":1.0},\"tol\":0.000001}";
        let mut env: Env = Env::new();
        env.insert(0, 0.4);
        env.insert(1, 0.6);
        assert_eq!(expr(source).eval(&env), Tri::True);
        env.insert(1, 0.7);
        assert_eq!(expr(source).eval(&env), Tri::False);
    }

    #[test]
    fn unknown_stays_unknown() {
        let source = "{\"t\":\"cmp\",\"op\":\"ge\",\"a\":{\"t\":\"var\",\"id\":9},\"b\":{\"t\":\"const\",\"v\":0.0}}";
        assert_eq!(expr(source).eval(&Env::new()), Tri::Unknown);
        let unsupported = "{\"t\":\"forall\"}";
        assert_eq!(expr(unsupported).eval(&Env::new()), Tri::Unknown);
    }

    #[test]
    fn kleene_logic_is_conservative() {
        assert_eq!(Tri::False.and(Tri::Unknown), Tri::False);
        assert_eq!(Tri::True.and(Tri::Unknown), Tri::Unknown);
        assert_eq!(Tri::True.or(Tri::Unknown), Tri::True);
        assert_eq!(Tri::Unknown.not(), Tri::Unknown);
        assert_eq!(Tri::Unknown.as_bool(), None);
    }

    #[test]
    fn variables_are_collected_in_order() {
        let source = "{\"t\":\"add\",\"a\":{\"t\":\"var\",\"id\":3},\"b\":{\"t\":\"var\",\"id\":1}}";
        let mut vars = Vec::new();
        expr(source).collect_vars(&mut vars);
        assert_eq!(vars, vec![3, 1]);
    }
}
