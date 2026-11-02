//! Soundness properties: the tests that would catch a *wrong* verdict.
//!
//! Example tests pin behaviour; these pin the contract that matters. Everything the engine
//! reports as a refutation must be unfalsifiable by sampling, every model it hands out must
//! satisfy its own constraints, and the two independent JSON recognizers must agree on every
//! string. A regression that makes the decoder prune a live branch — the expensive kind of bug —
//! shows up here as a single sampled counter-example.
//!
//! The generator is a deterministic xorshift PRNG rather than a property-testing crate,
//! because `veritas-core` has no dependencies at all: reproducible seeds are worth more than a
//! shrinking heuristics, and a failure prints the exact seed to replay.

use std::collections::BTreeMap;

use veritas_core::contractor::{Constraint, Contractor};
use veritas_core::expr::{CmpOp, Env, Expr};
use veritas_core::interval::Interval;
use veritas_core::json::{self, Json, Prefix};
use veritas_core::num::Rational;
use veritas_core::pda::Pda;
use veritas_core::solver::{self, Linear, Outcome};
use veritas_core::token_healer::{ByteTokenizer, TokenHealer, Tokenizer};
use veritas_core::wire;

/// xorshift64*: tiny, deterministic, good enough for property tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, limit: usize) -> usize {
        if limit == 0 {
            0
        } else {
            (self.next_u64() % limit as u64) as usize
        }
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// A linear or quadratic atom, kept in a shape the sampler can evaluate directly.
#[derive(Clone)]
enum Atom {
    Linear {
        coeffs: Vec<(u32, f64)>,
        rhs: f64,
        op: CmpOp,
    },
    Square {
        var: u32,
        rhs: f64,
        op: CmpOp,
    },
    Invalid,
}

impl Atom {
    fn satisfied(&self, point: &[f64], slack: f64) -> bool {
        match self {
            Atom::Linear { coeffs, rhs, op } => {
                let value: f64 = coeffs
                    .iter()
                    .map(|(var, coeff)| coeff * point[*var as usize])
                    .sum();
                holds(value, *rhs, *op, slack)
            }
            Atom::Square { var, rhs, op } => {
                let value = point[*var as usize] * point[*var as usize];
                holds(value, *rhs, *op, slack)
            }
            Atom::Invalid => true,
        }
    }

    fn constraint(&self) -> Option<Constraint> {
        match self {
            Atom::Linear { coeffs, rhs, op } => Some(Constraint::Linear {
                coeffs: coeffs.clone(),
                rhs: *rhs,
                op: *op,
            }),
            Atom::Square { var, rhs, op } => Some(Constraint::Square {
                var: *var,
                rhs: *rhs,
                op: *op,
            }),
            Atom::Invalid => None,
        }
    }
}

fn holds(value: f64, rhs: f64, op: CmpOp, slack: f64) -> bool {
    let slack = slack * (1.0 + value.abs() + rhs.abs());
    match op {
        CmpOp::Eq => (value - rhs).abs() <= slack,
        CmpOp::Le => value <= rhs + slack,
        CmpOp::Lt => value < rhs + slack,
        CmpOp::Ge => value >= rhs - slack,
        CmpOp::Gt => value > rhs - slack,
        _ => true,
    }
}

const COEFFICIENTS: [f64; 6] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0];
const BOUNDS: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
const OPS: [CmpOp; 3] = [CmpOp::Eq, CmpOp::Le, CmpOp::Ge];

/// A refutation must never contain a solution.
#[test]
fn contractor_refutation_is_sound() {
    let seed = 0x00C0_FFEE_1234_5678u64;
    let mut rng = Rng::new(seed);
    let mut refutations = 0usize;

    for _ in 0..300 {
        let mut domains: BTreeMap<u32, Interval> = BTreeMap::new();
        let mut edges = [(0.0, 0.0); 2];
        for (var, slot) in edges.iter_mut().enumerate() {
            let lo = rng.range(-2.0, 0.0);
            let hi = lo + rng.range(0.05, 3.0);
            *slot = (lo, hi);
            domains.insert(var as u32, Interval::new(lo, hi));
        }
        let mut atoms: Vec<Atom> = Vec::new();
        for _ in 0..(1 + rng.below(3)) {
            if rng.below(5) == 0 {
                atoms.push(Atom::Square {
                    var: rng.below(2) as u32,
                    rhs: rng.pick(&BOUNDS),
                    op: rng.pick(&[CmpOp::Eq, CmpOp::Le]),
                });
                continue;
            }
            let mut coeffs: Vec<(u32, f64)> = Vec::new();
            for var in 0..2u32 {
                let coeff = rng.pick(&COEFFICIENTS);
                if coeff != 0.0 {
                    coeffs.push((var, coeff));
                }
            }
            if coeffs.is_empty() {
                atoms.push(Atom::Invalid);
                continue;
            }
            atoms.push(Atom::Linear {
                coeffs,
                rhs: rng.range(-2.0, 2.0),
                op: rng.pick(&OPS),
            });
        }

        let mut contractor = Contractor::new(domains);
        for atom in &atoms {
            if let Some(constraint) = atom.constraint() {
                contractor.push(constraint);
            }
        }
        let feasible = contractor.propagate(8);
        if feasible {
            continue;
        }
        refutations += 1;

        // Sampling is incomplete but it only ever has to falsify: one point is enough.
        for _ in 0..1500 {
            let point = [rng.range(edges[0].0, edges[0].1), rng.range(edges[1].0, edges[1].1)];
            let solution = atoms.iter().all(|atom| atom.satisfied(&point, 0.0));
            assert!(
                !solution,
                "seed {seed}: contractor refuted a box containing a satisfying point {point:?}"
            );
        }
    }

    assert!(
        refutations > 20,
        "the generator must produce refutations for this test to mean anything, got {refutations}"
    );
}

/// `UNSAT` is a proof: no sample may satisfy the system, and the core alone must stay infeasible.
#[test]
fn solver_unsat_is_never_wrong() {
    let seed = 0xDEAD_BEEF_0BAD_F00Du64;
    let mut rng = Rng::new(seed);
    let mut unsat = 0usize;
    let mut sat = 0usize;

    for _ in 0..300 {
        let variables = 1 + rng.below(3);
        let mut constraints: Vec<Linear> = Vec::new();
        for _ in 0..(1 + rng.below(4)) {
            let mut coeffs: Vec<(u32, f64)> = Vec::new();
            for var in 0..variables as u32 {
                let coeff = rng.pick(&COEFFICIENTS);
                if coeff != 0.0 {
                    coeffs.push((var, coeff));
                }
            }
            if coeffs.is_empty() {
                continue;
            }
            constraints.push(Linear {
                coeffs,
                rhs: rng.pick(&BOUNDS),
                op: rng.pick(&OPS),
            });
        }
        if constraints.is_empty() {
            continue;
        }

        match solver::solve(&constraints) {
            Outcome::Sat { model, .. } => {
                sat += 1;
                let mut point = vec![0.0; variables];
                for (var, value) in &model {
                    if (*var as usize) < variables {
                        point[*var as usize] = *value;
                    }
                }
                for constraint in &constraints {
                    let atom = Atom::Linear {
                        coeffs: constraint.coeffs.clone(),
                        rhs: constraint.rhs,
                        op: constraint.op,
                    };
                    assert!(
                        atom.satisfied(&point, 1e-9),
                        "seed {seed}: model {point:?} violates {constraint:?}"
                    );
                }
            }
            Outcome::Unsat { core } => {
                unsat += 1;
                let subset: Vec<Linear> = core
                    .iter()
                    .filter_map(|index| constraints.get(*index).cloned())
                    .collect();
                assert!(!subset.is_empty(), "seed {seed}: an empty core is not a refutation");
                assert!(
                    matches!(solver::solve(&subset), Outcome::Unsat { .. }),
                    "seed {seed}: the reported core {core:?} is not infeasible on its own"
                );
                for point in grid(variables, 0.5) {
                    let solution = constraints.iter().all(|constraint| {
                        Atom::Linear {
                            coeffs: constraint.coeffs.clone(),
                            rhs: constraint.rhs,
                            op: constraint.op,
                        }
                        .satisfied(&point, 0.0)
                    });
                    assert!(
                        !solution,
                        "seed {seed}: UNSAT but {point:?} satisfies {constraints:?}"
                    );
                }
            }
            Outcome::Unknown { .. } => {}
        }
    }

    assert!(unsat > 20, "expected refutations, got {unsat}");
    assert!(sat > 5, "expected models, got {sat}");
}

/// Every point of a coarse grid in `[-2, 2]^variables`.
fn grid(variables: usize, step: f64) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = vec![Vec::new()];
    for _ in 0..variables {
        let mut next: Vec<Vec<f64>> = Vec::new();
        for point in &points {
            let mut value = -2.0;
            while value <= 2.0 + 1e-12 {
                let mut extended = point.clone();
                extended.push(value);
                next.push(extended);
                value += step;
            }
        }
        points = next;
    }
    points
}

/// With one variable the engine must be *complete*, not merely sound: strictness decides.
#[test]
fn one_variable_feasibility_is_complete() {
    let seed = 0x5EED_1BAD_2BAD_3BADu64;
    let mut rng = Rng::new(seed);
    let mut checked = 0usize;
    let mut decided = 0usize;

    for _ in 0..500 {
        let mut lo = rng.pick(&BOUNDS);
        let mut hi = rng.pick(&BOUNDS);
        let lo_strict = rng.below(2) == 0;
        let hi_strict = rng.below(2) == 0;
        if rng.below(2) == 0 {
            std::mem::swap(&mut lo, &mut hi);
        }
        let constraints = vec![
            Linear {
                coeffs: vec![(0, 1.0)],
                rhs: lo,
                op: if lo_strict { CmpOp::Gt } else { CmpOp::Ge },
            },
            Linear {
                coeffs: vec![(0, 1.0)],
                rhs: hi,
                op: if hi_strict { CmpOp::Lt } else { CmpOp::Le },
            },
        ];
        let expected = if lo < hi {
            true
        } else if lo > hi {
            false
        } else {
            !(lo_strict || hi_strict)
        };
        match solver::solve(&constraints) {
            Outcome::Sat { .. } => {
                decided += 1;
                assert!(expected, "seed {seed}: [{lo}, {hi}] reported SAT");
            }
            Outcome::Unsat { .. } => {
                decided += 1;
                assert!(!expected, "seed {seed}: [{lo}, {hi}] reported UNSAT");
            }
            Outcome::Unknown { .. } => {}
        }
        checked += 1;
    }
    assert_eq!(checked, 500);
    assert!(
        decided > 250,
        "a complete one-variable solver must decide most boxes, decided {decided}"
    );
}

/// JSON-legal ways to keep typing after `literal` (RFC 8259: no leading zeros).
fn legal_completions(literal: &str, allow_exp: bool, rng: &mut Rng) -> Vec<String> {
    let mut out = vec![literal.to_string()];
    let has_fraction = literal.contains('.');
    let zero_integer = literal == "0" || literal == "-0";
    if has_fraction {
        // Only more fraction digits may follow.
        for _ in 0..6 {
            out.push(format!("{literal}{}", rng.below(10)));
        }
        return out;
    }
    if !zero_integer {
        for _ in 0..4 {
            out.push(format!("{literal}{}", rng.below(10)));
        }
    }
    for _ in 0..3 {
        out.push(format!("{literal}.{}{}", rng.below(10), rng.below(10)));
    }
    if allow_exp {
        for _ in 0..2 {
            out.push(format!("{literal}e{}", 2 + rng.below(3)));
        }
    }
    out
}

/// The numeric prefix mask must contain every value the typed literal can still become.
#[test]
fn numeric_prefix_mask_never_drops_a_reachable_value() {
    let seed = 0x1234_ABCD_5555_7777u64;
    let mut rng = Rng::new(seed);
    let mut non_empty = 0usize;

    for _ in 0..600 {
        let negative = rng.below(2) == 0;
        let integer = rng.below(1000);
        let digits = rng.below(3);
        let mut literal = format!("{}{}", if negative { "-" } else { "" }, integer);
        if digits > 0 {
            literal.push('.');
            for _ in 0..digits {
                literal.push_str(&format!("{}", rng.below(10)));
            }
        }
        // Bounds are built *around* the literal: uniform bounds in a fixed window would mostly
        // miss it, and an exhaustive-mask test whose samples are all unreachable proves nothing.
        let center: f64 = literal.parse().expect("the generator only emits number prefixes");
        let width = rng.pick(&[0.5, 2.0, 20.0, 200.0]);
        let lo = center - width * rng.range(0.1, 0.9);
        let hi = center + width * rng.range(0.1, 0.9);
        let allow_exp = rng.below(2) == 0;
        let bounds = Interval::new(lo, hi);
        let span = Interval::from_numeric_prefix(&literal, bounds, allow_exp);

        // Every *JSON-legal* completion of the literal is a document the decoder may still
        // produce. Leading zeros are not legal, so "0" may only grow a fraction or exponent.
        let completions = legal_completions(&literal, allow_exp, &mut rng);
        let mut reachable = 0usize;
        for completion in &completions {
            let value: f64 = match completion.parse() {
                Ok(value) => value,
                Err(_) => continue,
            };
            if value < lo || value > hi {
                continue;
            }
            reachable += 1;
            assert!(
                !span.is_empty(),
                "seed {seed}: {completion} = {value} is inside {bounds:?} but the mask is empty"
            );
            assert!(
                span.contains(value),
                "seed {seed}: {completion} = {value} is inside {bounds:?} but outside {span:?}"
            );
        }
        if reachable > 0 {
            non_empty += 1;
        }
    }
    assert!(non_empty > 100, "expected reachable prefixes, got {non_empty}");
}

/// Two independent recognizers, one language: they must never disagree.
#[test]
fn the_automaton_and_the_parser_agree_on_random_strings() {
    let seed = 0xFEED_FACE_0BAD_C0DEu64;
    let mut rng = Rng::new(seed);
    let alphabet: Vec<char> = "{}[]\":,01.-e+tru falsen\\".chars().collect();

    for _ in 0..4000 {
        let length = rng.below(9);
        let mut text = String::new();
        for _ in 0..length {
            text.push(rng.pick(&alphabet));
        }
        let mut pda = Pda::new();
        let alive = pda.push_str(&text);
        let expected = json::prefix_status(&text);
        let actual = if pda.accepts() {
            Prefix::Complete
        } else if alive && pda.is_alive() {
            Prefix::Incomplete
        } else {
            Prefix::Invalid
        };
        assert_eq!(actual, expected, "seed {seed}: prefix {text:?}");
    }
}

/// Token healing may only report ids it verified.
#[test]
fn healing_never_reports_an_unverified_splice() {
    let seed = 0x7777_3333_1111_9999u64;
    let mut rng = Rng::new(seed);
    let tokenizer = ByteTokenizer;
    let healer = TokenHealer::new(3);
    let prefix_alphabet: Vec<char> = "{\"a\": 0123456789.,}[] ntrue".chars().collect();
    let splice_alphabet: Vec<char> = "0123456789.,\" }:ab".chars().collect();

    for _ in 0..400 {
        let mut prefix = String::new();
        for _ in 0..rng.below(12) {
            prefix.push(rng.pick(&prefix_alphabet));
        }
        let mut splice = String::new();
        for _ in 0..(1 + rng.below(8)) {
            splice.push(rng.pick(&splice_alphabet));
        }
        let prefill = healer.heal(&tokenizer, &prefix, &splice, None);
        assert!(prefill.backup_tokens <= 3, "backup must stay inside the budget");
        assert_eq!(prefill.splice_string, splice);
        assert_eq!(prefill.healed_text, format!("{prefix}{splice}"));
        if prefill.exact {
            let tail: String = prefix.chars().skip(prefill.offset).collect();
            let expected = format!("{tail}{splice}");
            assert_eq!(
                tokenizer.decode(&prefill.new_token_ids),
                expected,
                "seed {seed}: healed ids do not decode to the rewound tail plus splice"
            );
        } else {
            assert_eq!(prefill.new_token_ids, tokenizer.encode(&splice));
            assert_eq!(prefill.backup_tokens, 0);
        }
    }
}

/// Rendering then parsing must be the identity, including the number grammar.
#[test]
fn rendering_and_parsing_are_inverse() {
    let seed = 0x5555_4444_3333_2222u64;
    let mut rng = Rng::new(seed);

    for _ in 0..500 {
        let value = random_json(&mut rng, 0);
        let rendered = value.render();
        let parsed = json::parse(&rendered)
            .unwrap_or_else(|error| panic!("seed {seed}: {rendered} does not parse: {error}"));
        assert_eq!(parsed, value, "seed {seed}: round trip changed {rendered}");
    }
}

fn random_json(rng: &mut Rng, depth: usize) -> Json {
    let choice = if depth >= 3 { rng.below(4) } else { rng.below(6) };
    match choice {
        0 => Json::Null,
        1 => Json::Bool(rng.below(2) == 0),
        2 => Json::Number(rng.pick(&[0.0, -0.0, 1.0, -1.5, 1e-7, 1234.5, 1e16, -3.25])),
        3 => {
            let alphabet: Vec<char> = "abc \\\"\n\t\u{00e9}\u{1F600}".chars().collect();
            let mut text = String::new();
            for _ in 0..rng.below(6) {
                text.push(rng.pick(&alphabet));
            }
            Json::Str(text)
        }
        4 => {
            let mut items = Vec::new();
            for _ in 0..rng.below(3) {
                items.push(random_json(rng, depth + 1));
            }
            Json::Array(items)
        }
        _ => {
            let mut entries = Vec::new();
            for index in 0..rng.below(3) {
                entries.push((format!("k{}", index), random_json(rng, depth + 1)));
            }
            Json::Object(entries)
        }
    }
}

/// Expressions the wire format does not understand must answer `None`, never `0.0`.
#[test]
fn unsupported_expressions_are_unknown_not_zero() {
    let mut env: Env = Env::new();
    env.insert(0, 0.25);
    let missing = Expr::from_json(&json::parse("{\"t\":\"var\",\"id\":7}").unwrap());
    assert_eq!(missing.eval_number(&env), None);
    let unsupported = Expr::from_json(&json::parse("{\"t\":\"quantum\"}").unwrap());
    assert_eq!(unsupported.eval_number(&env), None);
    // Division by zero is not a number, so the answer is "cannot decide".
    let by_zero = Expr::from_json(
        &json::parse("{\"t\":\"div\",\"a\":{\"t\":\"var\",\"id\":0},\"b\":{\"t\":\"const\",\"v\":0.0}}")
            .unwrap(),
    );
    assert_eq!(by_zero.eval_number(&env), None);
}

/// Rationals are exact or absent: never quietly rounded.
#[test]
fn rationals_stay_exact_or_fail_closed() {
    let mut rng = Rng::new(0x0A0B_0C0D_0E0F_1011);
    for _ in 0..200 {
        let numerator = rng.range(-1e6, 1e6) as i128;
        let denominator = 1 + rng.below(999) as i128;
        let value = Rational::new(numerator, denominator).expect("representable");
        let doubled = value.add(value).expect("no overflow");
        assert_eq!(doubled, Rational::new(numerator * 2, denominator).unwrap());
        assert_eq!(value.neg().neg(), value);
        if !value.is_zero() {
            let reciprocal = Rational::new(1, 1).unwrap().div(value).expect("non-zero");
            assert_eq!(reciprocal.mul(value).unwrap(), Rational::ONE);
        }
    }
    assert!(Rational::new(1, 0).is_none());
}

/// The domain report must describe the machine that produced it, bit for bit — including the
/// intervals JSON cannot print as numbers. A caller that reacts to the report (`status:
/// "empty"`, a narrowed bound) acts on values that crossed a text boundary.
#[test]
fn domain_reports_round_trip_through_the_wire() {
    let seed = 0x0BAD_C0DE_5EED_1234u64;
    let mut rng = Rng::new(seed);
    let mut special = 0usize;

    for _ in 0..200 {
        let mut domains: BTreeMap<u32, Interval> = BTreeMap::new();
        for var in 0..(1 + rng.below(4)) as u32 {
            let interval = match rng.below(8) {
                0 => {
                    special += 1;
                    Interval::EMPTY
                }
                1 => {
                    special += 1;
                    Interval::new(f64::NEG_INFINITY, rng.range(-1.0, 1.0))
                }
                2 => {
                    special += 1;
                    Interval::new(rng.range(-1.0, 1.0), f64::INFINITY)
                }
                _ => {
                    let lo = rng.range(-4.0, 4.0);
                    Interval::new(lo, lo + rng.range(0.0, 4.0))
                }
            };
            domains.insert(var, interval);
        }

        let rendered = wire::domains_json(&Contractor::new(domains.clone())).render();
        let parsed = json::parse(&rendered).expect("the wire form is valid JSON");
        let restored = wire::parse_domains(&parsed).expect("the wire form parses back");
        assert_eq!(restored.len(), domains.len(), "seed {seed}: {rendered}");
        for (var, interval) in &domains {
            let back = restored.get(var).expect("every variable survives");
            assert_eq!(
                (back.lo.to_bits(), back.hi.to_bits()),
                (interval.lo.to_bits(), interval.hi.to_bits()),
                "seed {seed}: var {var} {interval:?} became {back:?} through {rendered}"
            );
        }
    }
    assert!(special > 20, "the generator must exercise unbounded domains");
}
