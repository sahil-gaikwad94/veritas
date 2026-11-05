//! BPE boundary healing and Jump-Forward prefill splicing.
//!
//! Port of `python/vexx/token_healer.py`. Appending a deterministic continuation
//! (`", \"sharpe_ratio\": 0.64367"`) to a token prefix produces a *different* tokenization than
//! the model would have produced at that boundary, so the KV cache and the text disagree.
//! Healing rewinds a few trailing tokens, re-tokenizes the rewound span together with the
//! splice, and reports the ids to prefill in one batch.
//!
//! The module is tokenizer-agnostic ([`Tokenizer`]) and ships [`ByteTokenizer`], which is
//! exact and dependency-free. When no exact boundary exists inside the backup budget the
//! result says `exact: false` — it never claims a healed splice it did not verify.

/// Minimal tokenizer surface: encode text to ids and decode ids back to text.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
}

/// UTF-8 byte tokenizer: one token per byte, exact and stable for tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|byte| *byte as u32).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|id| (*id & 0xFF) as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Instruction for a batched KV-cache extension.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JumpForwardPrefill {
    /// Tokens to rewind from the KV cache before appending `new_token_ids`.
    pub backup_tokens: usize,
    /// Ids to append; they normally decode to `rewound_tail + splice_string`.
    pub new_token_ids: Vec<u32>,
    pub splice_string: String,
    pub healed_text: String,
    /// Character offset in the prefix where the rewound tail starts.
    pub offset: usize,
    /// Whether the reported ids were verified to decode to exactly the expected text.
    pub exact: bool,
}

impl JumpForwardPrefill {
    pub fn num_new_tokens(&self) -> usize {
        self.new_token_ids.len()
    }
}

#[derive(Clone, Debug)]
pub struct TokenHealer {
    max_backup: usize,
}

impl Default for TokenHealer {
    fn default() -> TokenHealer {
        TokenHealer::new(4)
    }
}

impl TokenHealer {
    pub fn new(max_backup: usize) -> TokenHealer {
        TokenHealer {
            max_backup: max_backup.max(1),
        }
    }

    pub fn max_backup(&self) -> usize {
        self.max_backup
    }

    /// Find the smallest token backup that makes `splice` tokenize exactly.
    pub fn heal(
        &self,
        tokenizer: &dyn Tokenizer,
        prefix_text: &str,
        splice_string: &str,
        max_backup: Option<usize>,
    ) -> JumpForwardPrefill {
        if splice_string.is_empty() {
            return JumpForwardPrefill {
                backup_tokens: 0,
                new_token_ids: Vec::new(),
                splice_string: String::new(),
                healed_text: prefix_text.to_string(),
                offset: prefix_text.chars().count(),
                exact: true,
            };
        }
        let prefix_ids = tokenizer.encode(prefix_text);
        let full_text = format!("{}{}", prefix_text, splice_string);
        let full_ids = tokenizer.encode(&full_text);
        let budget = max_backup.unwrap_or(self.max_backup);

        for backup in 0..=budget.min(prefix_ids.len()) {
            let (cut_text, kept_ids) = if backup == 0 {
                (prefix_text.to_string(), prefix_ids.clone())
            } else {
                let kept = &prefix_ids[..prefix_ids.len() - backup];
                let cut_text = tokenizer.decode(kept);
                if !prefix_text.starts_with(&cut_text) {
                    continue;
                }
                let kept_ids = tokenizer.encode(&cut_text);
                // The retained prefix must tokenize identically, or the cache is still wrong.
                if kept_ids.len() > prefix_ids.len() || prefix_ids[..kept_ids.len()] != kept_ids[..]
                {
                    continue;
                }
                (cut_text, kept_ids)
            };
            if kept_ids.len() > full_ids.len() || full_ids[..kept_ids.len()] != kept_ids[..] {
                continue;
            }
            let new_ids: Vec<u32> = full_ids[kept_ids.len()..].to_vec();
            let expected = format!("{}{}", &prefix_text[cut_text.len()..], splice_string);
            if tokenizer.decode(&new_ids) == expected {
                return JumpForwardPrefill {
                    backup_tokens: backup,
                    new_token_ids: new_ids,
                    splice_string: splice_string.to_string(),
                    healed_text: full_text,
                    offset: cut_text.chars().count(),
                    exact: true,
                };
            }
        }

        // No exact boundary inside the budget: a usable prefill, but no healing claim.
        JumpForwardPrefill {
            backup_tokens: 0,
            new_token_ids: tokenizer.encode(splice_string),
            splice_string: splice_string.to_string(),
            healed_text: full_text,
            offset: prefix_text.chars().count(),
            exact: false,
        }
    }

    /// [`TokenHealer::heal`] that refuses the unverified fallback.
    pub fn heal_exact(
        &self,
        tokenizer: &dyn Tokenizer,
        prefix_text: &str,
        splice_string: &str,
        max_backup: Option<usize>,
    ) -> Option<JumpForwardPrefill> {
        let prefill = self.heal(tokenizer, prefix_text, splice_string, max_backup);
        if prefill.exact {
            Some(prefill)
        } else {
            None
        }
    }

    /// Heal from a token prefix: decode it, then heal the text.
    pub fn heal_ids(
        &self,
        tokenizer: &dyn Tokenizer,
        prefix_ids: &[u32],
        splice_string: &str,
    ) -> JumpForwardPrefill {
        let prefix_text = tokenizer.decode(prefix_ids);
        self.heal(tokenizer, &prefix_text, splice_string, None)
    }

    /// Character-level compatibility API for callers that have no tokenizer.
    pub fn heal_and_splice_chars(&self, prefix: &[String], splice_string: &str) -> JumpForwardPrefill {
        let mut backup = 0usize;
        if let Some(last) = prefix.last() {
            if !last.is_empty() && splice_string.starts_with(last.as_str()) {
                backup = 1;
            }
        }
        JumpForwardPrefill {
            backup_tokens: backup,
            new_token_ids: splice_string.chars().map(|ch| ch as u32).collect(),
            splice_string: splice_string.to_string(),
            healed_text: format!("{}{}", prefix.concat(), splice_string),
            offset: prefix.len(),
            exact: true,
        }
    }

    /// Splice a pre-rendered 0-DoF continuation after `last_token_id`.
    pub fn compute_token_healed_splice(
        &self,
        tokenizer: &dyn Tokenizer,
        splice_string: &str,
        last_token_id: Option<u32>,
    ) -> JumpForwardPrefill {
        match last_token_id {
            Some(id) => self.heal_ids(tokenizer, &[id], splice_string),
            None => self.heal(tokenizer, "", splice_string, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately broken tokenizer: it drops the final byte of every string, so the
    /// re-tokenized prefix can never match and healing must refuse to claim exactness.
    struct LossyTokenizer;

    impl Tokenizer for LossyTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            let bytes = text.as_bytes();
            bytes[..bytes.len().saturating_sub(1)]
                .iter()
                .map(|byte| *byte as u32)
                .collect()
        }

        fn decode(&self, ids: &[u32]) -> String {
            ByteTokenizer.decode(ids)
        }
    }

    #[test]
    fn byte_tokenizer_round_trips() {
        let tokenizer = ByteTokenizer;
        let text = "{\"a\": 1}\n";
        assert_eq!(tokenizer.decode(&tokenizer.encode(text)), text);
        assert_eq!(tokenizer.encode("ab"), vec![97, 98]);
    }

    #[test]
    fn healing_verifies_the_splice_text() {
        let healer = TokenHealer::new(4);
        let tokenizer = ByteTokenizer;
        let prefix = "{\"a\": 0.6";
        let splice = ", \"b\": 1}";
        let prefill = healer.heal(&tokenizer, prefix, splice, None);
        assert!(prefill.exact);
        assert_eq!(prefill.healed_text, format!("{}{}", prefix, splice));
        assert_eq!(tokenizer.decode(&prefill.new_token_ids), splice);
        assert_eq!(prefill.num_new_tokens(), splice.len());
        assert!(healer
            .heal_exact(&tokenizer, prefix, splice, None)
            .is_some());
    }

    #[test]
    fn empty_splice_is_a_no_op() {
        let healer = TokenHealer::default();
        let prefill = healer.heal(&ByteTokenizer, "{\"a\": 1", "", None);
        assert!(prefill.exact && prefill.new_token_ids.is_empty());
        assert_eq!(prefill.healed_text, "{\"a\": 1");
    }

    #[test]
    fn a_lossy_tokenizer_never_claims_healing() {
        let healer = TokenHealer::new(4);
        let prefill = healer.heal(&LossyTokenizer, "{\"a\": 1", ", \"b\": 2}", None);
        assert!(!prefill.exact);
        assert_eq!(prefill.backup_tokens, 0);
        assert!(healer
            .heal_exact(&LossyTokenizer, "{\"a\": 1", ", \"b\": 2}", None)
            .is_none());
    }

    #[test]
    fn id_prefixes_heal_through_the_tokenizer() {
        let healer = TokenHealer::default();
        let tokenizer = ByteTokenizer;
        let ids = tokenizer.encode("{\"a\": 0.6");
        let prefill = healer.heal_ids(&tokenizer, &ids, ", \"b\": 1}");
        assert!(prefill.exact);
        let single = healer.compute_token_healed_splice(&tokenizer, ", \"b\": 1}", Some(125));
        assert!(single.exact);
        assert_eq!(single.healed_text, "}, \"b\": 1}");
    }

    #[test]
    fn character_level_compatibility_api() {
        let healer = TokenHealer::default();
        let prefill = healer.heal_and_splice_chars(&["{\"a\": 0.6".to_string()], ", \"b\": 1}");
        assert_eq!(prefill.backup_tokens, 0);
        assert_eq!(prefill.healed_text, "{\"a\": 0.6, \"b\": 1}");
        let starting_with_last = healer.heal_and_splice_chars(&["ab".to_string()], "abc");
        assert_eq!(starting_with_last.backup_tokens, 1);
    }
}
