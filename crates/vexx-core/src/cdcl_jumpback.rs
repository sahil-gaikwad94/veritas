//! Entropy-weighted CDCL jumpback, tabu prefixes and half-space domain cuts.
//!
//! Port of `python/vexx/cdcl_jumpback.py`. When the solver refutes a partial document, the
//! decoder must decide *how far back* to rewind. Rewinding to the immediately preceding
//! clause is the standard, wasteful answer; this module scores each clause in the unsat core
//! by how much uncertainty it carried (mean token entropy) and how often it has already
//! conflicted, then rewinds to the single most culpable one.
//!
//! Every jumpback also emits a [`ConflictBranchRecord`] so `vexx.rl` can build process-level
//! preference pairs out of the rejected branch.

/// One committed clause, recorded as the decoder advances.
#[derive(Clone, Debug, Default)]
pub struct ClauseCheckpoint {
    pub clause_id: i64,
    pub var_id: u32,
    pub committed_val: f64,
    pub start_token_idx: usize,
    pub end_token_idx: usize,
    pub mean_shannon_entropy: f64,
    pub token_sequence: Vec<u32>,
    pub conflict_count: u32,
}

/// A rejected branch, exported for reinforcement learning.
#[derive(Clone, Debug, Default)]
pub struct ConflictBranchRecord {
    pub prefix_tokens: Vec<u32>,
    pub rejected_clause_tokens: Vec<u32>,
    pub unsat_core_ids: Vec<i64>,
    pub violation_margin: f64,
}

/// Which half-space a domain cut removes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutDirection {
    /// Keep `[lo, committed_val - tol]`.
    Upper,
    /// Keep `[committed_val + tol, hi]`.
    Lower,
}

impl CutDirection {
    pub fn parse(text: &str) -> CutDirection {
        match text {
            "lower" => CutDirection::Lower,
            _ => CutDirection::Upper,
        }
    }
}

/// The domain narrowing a jumpback asks the contractor to apply.
#[derive(Clone, Debug, PartialEq)]
pub struct DomainCut {
    pub var_id: u32,
    pub committed_val: f64,
    pub depth: i64,
    pub direction: CutDirection,
}

/// What the decoder should do next.
#[derive(Clone, Debug, Default)]
pub struct Jumpback {
    pub target_clause: Option<i64>,
    pub rewind_to: Option<usize>,
    pub tabu_added: bool,
    pub domain_cut: Option<DomainCut>,
}

/// Set of token sequences that must not be sampled again at a given position.
#[derive(Clone, Debug, Default)]
pub struct TabuPrefixTrie {
    entries: std::collections::BTreeMap<usize, std::collections::BTreeSet<Vec<u32>>>,
}

impl TabuPrefixTrie {
    pub fn add(&mut self, start_index: usize, sequence: &[u32]) {
        self.entries
            .entry(start_index)
            .or_default()
            .insert(sequence.to_vec());
    }

    pub fn contains(&self, start_index: usize, sequence: &[u32]) -> bool {
        self.entries
            .get(&start_index)
            .map(|set| set.contains(sequence))
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.entries.values().map(|set| set.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct CdclJumpbackEngine {
    lambda: f64,
    gamma: f64,
    checkpoints: Vec<ClauseCheckpoint>,
    tabu: TabuPrefixTrie,
    domain_cuts: Vec<DomainCut>,
    records: Vec<ConflictBranchRecord>,
    tolerance: f64,
}

impl Default for CdclJumpbackEngine {
    fn default() -> CdclJumpbackEngine {
        CdclJumpbackEngine::new(0.5, 0.1)
    }
}

impl CdclJumpbackEngine {
    pub fn new(lambda: f64, gamma: f64) -> CdclJumpbackEngine {
        CdclJumpbackEngine {
            lambda,
            gamma,
            checkpoints: Vec::new(),
            tabu: TabuPrefixTrie::default(),
            domain_cuts: Vec::new(),
            records: Vec::new(),
            tolerance: 1e-9,
        }
    }

    pub fn push_checkpoint(&mut self, checkpoint: ClauseCheckpoint) {
        self.checkpoints.push(checkpoint);
    }

    pub fn checkpoints(&self) -> &[ClauseCheckpoint] {
        &self.checkpoints
    }

    pub fn tabu(&self) -> &TabuPrefixTrie {
        &self.tabu
    }

    pub fn domain_cuts(&self) -> &[DomainCut] {
        &self.domain_cuts
    }

    pub fn records(&self) -> &[ConflictBranchRecord] {
        &self.records
    }

    /// `score(c) = entropy(c) * exp(-lambda * (current - c)) + gamma * conflicts(c)`.
    pub fn culprit_score(&self, checkpoint: &ClauseCheckpoint, current_clause: i64) -> f64 {
        let distance = (current_clause - checkpoint.clause_id) as f64;
        checkpoint.mean_shannon_entropy * (-self.lambda * distance).exp()
            + self.gamma * checkpoint.conflict_count as f64
    }

    pub fn select_culprit(&self, unsat_core: &[i64]) -> Option<&ClauseCheckpoint> {
        let current = unsat_core.last().copied().unwrap_or(self.checkpoints.len() as i64);
        let mut best: Option<&ClauseCheckpoint> = None;
        let mut best_score = f64::NEG_INFINITY;
        for checkpoint in &self.checkpoints {
            if !unsat_core.contains(&checkpoint.clause_id) {
                continue;
            }
            let score = self.culprit_score(checkpoint, current);
            if best.is_none() || score > best_score {
                best = Some(checkpoint);
                best_score = score;
            }
        }
        if best.is_some() {
            return best;
        }
        // No overlap between the core and the recorded clauses: blame the last clause.
        self.checkpoints.last()
    }

    pub fn jumpback(&mut self, unsat_core: &[i64]) -> Jumpback {
        let culprit = match self.select_culprit(unsat_core) {
            Some(checkpoint) => checkpoint.clone(),
            None => return Jumpback::default(),
        };
        self.tabu.add(culprit.start_token_idx, &culprit.token_sequence);
        let cut = DomainCut {
            var_id: culprit.var_id,
            committed_val: culprit.committed_val,
            depth: culprit.clause_id,
            direction: CutDirection::Upper,
        };
        self.domain_cuts.push(cut.clone());
        self.records.push(ConflictBranchRecord {
            prefix_tokens: culprit.token_sequence.clone(),
            rejected_clause_tokens: culprit.token_sequence.clone(),
            unsat_core_ids: unsat_core.to_vec(),
            violation_margin: 0.0,
        });
        Jumpback {
            target_clause: Some(culprit.clause_id),
            rewind_to: Some(culprit.start_token_idx.saturating_sub(1)),
            tabu_added: true,
            domain_cut: Some(cut),
        }
    }

    /// Domain narrowing for the culprit: the half-space that contains the conflict.
    pub fn apply_domain_cut(
        &mut self,
        culprit: &ClauseCheckpoint,
        direction: CutDirection,
    ) -> (f64, f64) {
        self.tolerance = self.tolerance.max(0.0);
        match direction {
            CutDirection::Upper => (f64::NEG_INFINITY, culprit.committed_val - self.tolerance),
            CutDirection::Lower => (culprit.committed_val + self.tolerance, f64::INFINITY),
        }
    }

    pub fn export_records(&self) -> Vec<ConflictBranchRecord> {
        self.records.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(id: i64, entropy: f64, conflicts: u32, start: usize) -> ClauseCheckpoint {
        ClauseCheckpoint {
            clause_id: id,
            var_id: 7,
            committed_val: 1.25,
            start_token_idx: start,
            end_token_idx: start + 2,
            mean_shannon_entropy: entropy,
            token_sequence: vec![10, 11],
            conflict_count: conflicts,
        }
    }

    #[test]
    fn the_high_entropy_clause_is_the_culprit() {
        let mut engine = CdclJumpbackEngine::new(0.5, 0.1);
        engine.push_checkpoint(checkpoint(0, 0.01, 0, 0));
        engine.push_checkpoint(checkpoint(1, 0.9, 0, 4));
        engine.push_checkpoint(checkpoint(2, 0.2, 0, 8));
        let chosen = engine.select_culprit(&[0, 1, 2]).expect("a culprit");
        assert_eq!(chosen.clause_id, 1);
    }

    #[test]
    fn distance_decays_entropy_but_conflicts_accumulate() {
        let engine = CdclJumpbackEngine::new(0.5, 0.1);
        let early = checkpoint(0, 1.0, 0, 0);
        let late = checkpoint(9, 0.5, 0, 0);
        assert!(engine.culprit_score(&early, 10) < engine.culprit_score(&late, 10));

        let mut engine = CdclJumpbackEngine::new(0.5, 0.1);
        engine.push_checkpoint(checkpoint(0, 0.1, 5, 0));
        engine.push_checkpoint(checkpoint(1, 0.2, 0, 4));
        // gamma * 5 > the entropy difference, so the repeat offender is chosen.
        assert_eq!(engine.select_culprit(&[0, 1]).unwrap().clause_id, 0);
    }

    #[test]
    fn jumpback_records_tabu_cut_and_the_rl_record() {
        let mut engine = CdclJumpbackEngine::new(0.5, 0.1);
        engine.push_checkpoint(checkpoint(3, 0.7, 1, 5));
        let jump = engine.jumpback(&[3]);
        assert_eq!(jump.target_clause, Some(3));
        assert_eq!(jump.rewind_to, Some(4));
        assert!(jump.tabu_added);
        assert!(engine.tabu().contains(5, &[10, 11]));
        assert_eq!(engine.domain_cuts().len(), 1);
        assert_eq!(engine.export_records().len(), 1);
        assert_eq!(engine.export_records()[0].unsat_core_ids, vec![3]);
    }

    #[test]
    fn domain_cuts_remove_the_conflicting_half_space() {
        let mut engine = CdclJumpbackEngine::default();
        let culprit = checkpoint(0, 0.5, 0, 0);
        let (_, upper) = engine.apply_domain_cut(&culprit, CutDirection::Upper);
        assert!(upper < culprit.committed_val);
        let (lower, _) = engine.apply_domain_cut(&culprit, CutDirection::Lower);
        assert!(lower > culprit.committed_val);
    }

    #[test]
    fn an_empty_engine_reports_no_target() {
        let mut engine = CdclJumpbackEngine::default();
        let jump = engine.jumpback(&[1, 2]);
        assert!(jump.target_clause.is_none() && jump.rewind_to.is_none());
        assert!(!jump.tabu_added);
    }
}
