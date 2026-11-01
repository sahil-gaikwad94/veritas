// Phase 2 CRANE controller placeholder
pub enum ReasoningMode { Reasoning, Emit }
pub struct CraneModeController { pub mode: ReasoningMode }
impl CraneModeController {
    pub fn in_reasoning(&self) -> bool { matches!(self.mode, ReasoningMode::Reasoning) }
}
