//! CRANE dual-mode controller: an unconstrained `<think>` scratchpad, then enforcement.
//!
//! Port of `python/vexx/crane_mode.py`. While the model is inside the reasoning window the
//! decoder must *not* mask: the model is allowed to write whatever helps it think. The moment
//! `</think>` (or `<emit>`) appears, Tier 0/1/2 masking switches on for the emitted document.
//!
//! The controller also passively mines `name = number` lines from the scratchpad so those
//! values can pre-warm the tier-1 domains — the reason the blueprint calls this "CRANE".

/// Which side of the `think` window the decoder is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningMode {
    /// Unconstrained scratchpad: masks are bypassed.
    Reasoning,
    /// Emitting the verified document: masks are enforced.
    Emit,
}

/// A mode switch, in the order it happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CraneEvent {
    SwitchToEmit,
    SwitchToReasoning,
}

#[derive(Clone, Debug)]
pub struct CraneModeController {
    think_open: String,
    think_close: String,
    emit_open: String,
    emit_close: String,
    mode: ReasoningMode,
    buffer: String,
    mined_equations: Vec<String>,
    cursor: usize,
}

impl Default for CraneModeController {
    fn default() -> CraneModeController {
        CraneModeController::new()
    }
}

impl CraneModeController {
    pub fn new() -> CraneModeController {
        CraneModeController::with_tags("<think>", "</think>", "<emit>", "</emit>")
    }

    pub fn with_tags(
        think_open: &str,
        think_close: &str,
        emit_open: &str,
        emit_close: &str,
    ) -> CraneModeController {
        CraneModeController {
            think_open: think_open.to_string(),
            think_close: think_close.to_string(),
            emit_open: emit_open.to_string(),
            emit_close: emit_close.to_string(),
            mode: ReasoningMode::Reasoning,
            buffer: String::new(),
            mined_equations: Vec::new(),
            cursor: 0,
        }
    }

    pub fn mode(&self) -> ReasoningMode {
        self.mode
    }

    pub fn in_reasoning(&self) -> bool {
        self.mode == ReasoningMode::Reasoning
    }

    /// Should the decoder enforce masks on the next token?
    pub fn masking_active(&self) -> bool {
        self.mode == ReasoningMode::Emit
    }

    pub fn mined_equations(&self) -> &[String] {
        &self.mined_equations
    }

    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    pub fn reset(&mut self) {
        self.mode = ReasoningMode::Reasoning;
        self.buffer.clear();
        self.mined_equations.clear();
        self.cursor = 0;
    }

    /// Consume generated text and report the mode switches it contains.
    pub fn feed(&mut self, text: &str) -> Vec<CraneEvent> {
        self.buffer.push_str(text);
        let mut events = Vec::new();
        loop {
            // Find the earliest tag at or after the cursor. Only lengths escape this scan:
            // holding a &str borrowed from self would block the mode switch below.
            let mut best: Option<(usize, usize, Tag)> = None;
            for (tag, kind) in [
                (self.think_open.as_str(), Tag::Open),
                (self.think_close.as_str(), Tag::Close),
                (self.emit_open.as_str(), Tag::EmitOpen),
                (self.emit_close.as_str(), Tag::EmitClose),
            ] {
                if tag.is_empty() {
                    continue;
                }
                if let Some(offset) = self.buffer[self.cursor.min(self.buffer.len())..].find(tag) {
                    let index = self.cursor + offset;
                    let earlier = best.map(|(best_index, _, _)| index < best_index).unwrap_or(true);
                    if earlier {
                        best = Some((index, tag.len(), kind));
                    }
                }
            }
            let (index, tag_len, kind) = match best {
                Some(found) => found,
                None => break,
            };
            match (kind, self.mode) {
                (Tag::Close, ReasoningMode::Reasoning) | (Tag::EmitOpen, ReasoningMode::Reasoning) => {
                    self.mode = ReasoningMode::Emit;
                    events.push(CraneEvent::SwitchToEmit);
                    self.mine_buffer();
                }
                (Tag::Open, ReasoningMode::Emit) | (Tag::EmitClose, ReasoningMode::Emit) => {
                    self.mode = ReasoningMode::Reasoning;
                    events.push(CraneEvent::SwitchToReasoning);
                }
                _ => {}
            }
            self.cursor = index + tag_len;
        }
        events
    }

    /// Collect `name = value` candidates from text without changing the mode.
    pub fn passive_mine(&mut self, text: &str) -> Vec<String> {
        let mut found = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.contains('=') {
                found.push(trimmed.to_string());
                self.mined_equations.push(trimmed.to_string());
            }
        }
        found
    }

    fn mine_buffer(&mut self) {
        let lines: Vec<String> = self.buffer.lines().map(|line| line.to_string()).collect();
        for line in lines {
            if line.contains('=') {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    self.mined_equations.push(trimmed);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tag {
    Open,
    Close,
    EmitOpen,
    EmitClose,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_turn_on_after_the_think_window() {
        let mut controller = CraneModeController::new();
        assert!(controller.in_reasoning() && !controller.masking_active());
        assert!(controller.feed("<think>w1 = 0.1").is_empty());
        assert!(controller.in_reasoning());
        let events = controller.feed(" and w2 = 0.2</think>{");
        assert_eq!(events, vec![CraneEvent::SwitchToEmit]);
        assert!(controller.masking_active());
        // Mining is passive: the scratchpad numbers are captured on the switch.
        assert!(controller
            .mined_equations()
            .iter()
            .any(|line| line.contains("w1 = 0.1")));
    }

    #[test]
    fn explicit_emit_tags_switch_both_ways() {
        let mut controller = CraneModeController::new();
        assert_eq!(controller.feed("<emit>{" ), vec![CraneEvent::SwitchToEmit]);
        assert_eq!(
            controller.feed("}</emit>"),
            vec![CraneEvent::SwitchToReasoning]
        );
        assert!(controller.in_reasoning());
    }

    #[test]
    fn tags_split_across_chunks_are_still_detected() {
        let mut controller = CraneModeController::new();
        assert!(controller.feed("<thi").is_empty());
        let events = controller.feed("nk>plan 1 + 1</thi");
        assert!(events.is_empty());
        assert_eq!(controller.feed("nk>"), vec![CraneEvent::SwitchToEmit]);
    }

    #[test]
    fn passive_mining_and_reset() {
        let mut controller = CraneModeController::new();
        let found = controller.passive_mine("alpha = 1\nnoise\nbeta = 2.5\n");
        assert_eq!(found, vec!["alpha = 1", "beta = 2.5"]);
        controller.reset();
        assert!(controller.mined_equations().is_empty() && controller.buffer().is_empty());
    }
}
