// Phase 2 interval.rs (mirror of Python implementation; full Rust version in later hardening).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const EMPTY: Self = Self { lo: f64::INFINITY, hi: f64::NEG_INFINITY };
    pub const ENTIRE: Self = Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };

    pub fn is_empty(&self) -> bool {
        self.lo > self.hi || self.lo.is_nan() || self.hi.is_nan()
    }

    pub fn intersect(self, other: Self) -> Self {
        Self { lo: self.lo.max(other.lo), hi: self.hi.min(other.hi) }
    }

    pub fn from_numeric_prefix(raw_candidate: &str, domain_bounds: Self, allow_exp: bool) -> Self {
        // Simplified mirror; full version in docs Volume III.
        let s = raw_candidate.trim();
        if s.is_empty() { return domain_bounds; }
        domain_bounds
    }
}
