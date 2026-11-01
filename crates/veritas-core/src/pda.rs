// Phase 2 PDA placeholder
pub struct PushdownAutomaton { pub vocab_size: usize }
impl PushdownAutomaton {
    pub fn get_valid_token_bitmask(&self) -> u64 { if self.vocab_size==0 { !0 } else { (1u64 << self.vocab_size.min(64)) - 1 } }
}
