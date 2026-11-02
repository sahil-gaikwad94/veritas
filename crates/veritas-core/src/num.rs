//! Float stepping and exact rationals.
//!
//! Two things live here because everything else depends on them:
//!
//! * [`next_up`] / [`next_down`] — the outward rounding used by every interval operation.
//!   Without it a sound contractor reports feasible points as dead ends (`a + b == 1` with
//!   `a = 0.023` is the canonical example: `1 - 0.977` lands one ULP outside the point
//!   domain).
//! * [`Rational`] — the exact arithmetic the feasibility solver uses, so `UNSAT` is a
//!   *proof* rather than a rounded guess. Overflow, or a value that is not representable,
//!   yields `None` and the caller degrades to `UNKNOWN` (fail-closed).

/// Next representable `f64` above `x` (identity for `+inf` and NaN).
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    let stepped = if x > 0.0 { bits + 1 } else { bits - 1 };
    f64::from_bits(stepped)
}

/// Next representable `f64` below `x` (identity for `-inf` and NaN).
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    let stepped = if x > 0.0 { bits - 1 } else { bits + 1 };
    f64::from_bits(stepped)
}

/// Widen by one ULP on each side; used after any rounded interval operation.
pub fn outward(lo: f64, hi: f64) -> (f64, f64) {
    (next_down(lo), next_up(hi))
}

fn gcd(a: i128, b: i128) -> i128 {
    let mut x = a.unsigned_abs();
    let mut y = b.unsigned_abs();
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    if x == 0 {
        1
    } else {
        x as i128
    }
}

/// Exact rational number in lowest terms with a positive denominator.
///
/// The representation is `i128` numerator and denominator: wide enough for the folding the
/// feasibility solver performs on real schemas, still bounded. When an operation cannot be
/// represented the constructor returns `None`, which the solver treats as *unknown*. That is
/// the honest failure mode for an exact decision procedure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    num: i128,
    den: i128,
}

#[allow(clippy::should_implement_trait)] // fallible ops: `Option` cannot be `std::ops::Add`
impl Rational {
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    pub fn new(num: i128, den: i128) -> Option<Rational> {
        if den == 0 {
            return None;
        }
        let sign = if den < 0 { -1i128 } else { 1i128 };
        let numerator = num.checked_mul(sign)?;
        let denominator = den.checked_mul(sign)?;
        let divisor = gcd(numerator, denominator);
        Some(Rational {
            num: numerator / divisor,
            den: denominator / divisor,
        })
    }

    pub fn from_int(value: i64) -> Rational {
        Rational {
            num: value as i128,
            den: 1,
        }
    }

    /// Exact conversion of an `i128` (used when folding literals in the solver).
    pub fn from_i128(value: i128) -> Rational {
        Rational { num: value, den: 1 }
    }

    /// Exact conversion of an `f64` (every finite float is a dyadic rational).
    pub fn from_f64(value: f64) -> Option<Rational> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Rational::ZERO);
        }
        let bits = value.to_bits();
        let negative = (bits >> 63) == 1;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        let (mantissa, exponent) = if exponent_bits == 0 {
            (fraction, -1074i32)
        } else {
            (fraction | 0x0010_0000_0000_0000, exponent_bits - 1075)
        };
        let signed = if negative {
            -(mantissa as i128)
        } else {
            mantissa as i128
        };
        let (num, den) = if exponent >= 0 {
            let shift = u32::try_from(exponent).ok()?;
            if shift >= 127 {
                return None;
            }
            (signed.checked_shl(shift)?, 1i128)
        } else {
            let shift = u32::try_from(-exponent).ok()?;
            if shift >= 127 {
                return None;
            }
            (signed, 1i128.checked_shl(shift)?)
        };
        Rational::new(num, den)
    }

    pub fn numer(self) -> i128 {
        self.num
    }

    pub fn denom(self) -> i128 {
        self.den
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    pub fn is_positive(self) -> bool {
        self.num > 0
    }

    pub fn is_negative(self) -> bool {
        self.num < 0
    }

    pub fn neg(self) -> Rational {
        Rational {
            num: -self.num,
            den: self.den,
        }
    }

    pub fn add(self, other: Rational) -> Option<Rational> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        let num = left.checked_add(right)?;
        let den = self.den.checked_mul(other.den)?;
        Rational::new(num, den)
    }

    pub fn sub(self, other: Rational) -> Option<Rational> {
        self.add(other.neg())
    }

    pub fn mul(self, other: Rational) -> Option<Rational> {
        let num = self.num.checked_mul(other.num)?;
        let den = self.den.checked_mul(other.den)?;
        Rational::new(num, den)
    }

    pub fn div(self, other: Rational) -> Option<Rational> {
        if other.is_zero() {
            return None;
        }
        let num = self.num.checked_mul(other.den)?;
        let den = self.den.checked_mul(other.num)?;
        Rational::new(num, den)
    }

    pub fn lt(self, other: Rational) -> Option<bool> {
        Some(self.compare(other)? == std::cmp::Ordering::Less)
    }

    pub fn le(self, other: Rational) -> Option<bool> {
        Some(self.compare(other)? != std::cmp::Ordering::Greater)
    }

    pub fn max(self, other: Rational) -> Option<Rational> {
        match self.compare(other)? {
            std::cmp::Ordering::Less => Some(other),
            _ => Some(self),
        }
    }

    /// Round-trip an exact rational to `f64` without pretending to be exact.
    pub fn approx(self) -> f64 {
        self.to_f64()
    }

    /// Total order, exact when it fits and `None` on overflow.
    pub fn compare(self, other: Rational) -> Option<std::cmp::Ordering> {
        let left = self.num.checked_mul(other.den)?;
        let right = other.num.checked_mul(self.den)?;
        Some(left.cmp(&right))
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stepping_is_monotone_and_tight() {
        assert!(next_up(1.0) > 1.0);
        assert!(next_down(1.0) < 1.0);
        assert!(next_up(1.0) - 1.0 < 1e-15);
        assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
        assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(next_up(0.0) > 0.0);
        assert!(next_down(0.0) < 0.0);
        assert!(next_up(-2.0) > -2.0);
        assert!(next_down(-2.0) < -2.0);
        // Round-tripping through the neighbour returns the original value.
        assert_eq!(next_down(next_up(3.25)), 3.25);
    }

    #[test]
    fn rationals_are_exact() {
        let third = Rational::new(1, 3).expect("1/3");
        let sum = third.add(third).expect("2/3");
        let sum = sum.add(third).expect("1");
        assert_eq!(sum, Rational::ONE);
        assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
        assert_eq!(Rational::new(1, -2), Rational::new(-1, 2));
        assert_eq!(Rational::new(1, 0), None);
        let tenth = Rational::from_f64(0.1).expect("0.1 is dyadic");
        // 0.1 is not exactly one tenth: the float expansion is a different rational.
        assert_ne!(tenth, Rational::new(1, 10).unwrap());
        assert!(tenth.compare(Rational::new(1, 10).unwrap()).is_some());
        // i128 widening survives folding that would have overflowed an i64 representation.
        let mut acc = tenth;
        for _ in 0..8 { acc = acc.add(tenth).expect("no overflow"); }
        assert!(acc.compare(Rational::ONE).unwrap() == std::cmp::Ordering::Less);
    }

    #[test]
    fn comparison_and_signs() {
        let a = Rational::new(-3, 4).unwrap();
        let b = Rational::new(-1, 2).unwrap();
        assert!(a.compare(b) == Some(std::cmp::Ordering::Less));
        assert!(a.is_negative() && !a.is_positive());
        assert!(Rational::ZERO.is_zero());
        assert_eq!(a.neg().to_f64(), 0.75);
    }
}
