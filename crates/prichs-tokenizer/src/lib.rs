use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct PrichsTokenizer {
    hf_tokenizer: Option<Tokenizer>,
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub special_tokens: HashMap<String, u32>,
}

impl PrichsTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::new();

        // Core Byte Tokens (0..255)
        for b in 0..=255u8 {
            let s = format!("<0x{:02X}>", b);
            vocab.insert(s.clone(), b as u32);
            id_to_token.push(s);
        }

        let mut tokenizer = Self {
            hf_tokenizer: None,
            vocab,
            id_to_token,
            special_tokens: HashMap::new(),
        };

        tokenizer.add_special_token("<pad>");
        tokenizer.add_special_token("<bos>");
        tokenizer.add_special_token("<eos>");
        tokenizer.add_special_token("<unk>");
        tokenizer.add_special_token("<prichs_think>");
        tokenizer.add_special_token("</prichs_think>");
        tokenizer.add_special_token("<starkdb_query>");
        tokenizer.add_special_token("</starkdb_query>");
        tokenizer.add_special_token("<stark_action>");

        tokenizer
    }

    pub fn from_huggingface_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tok = Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer: {}", e))?;

        let vocab = tok.get_vocab(true);
        let mut id_to_token = vec![String::new(); vocab.len()];
        for (token, id) in &vocab {
            if (*id as usize) < id_to_token.len() {
                id_to_token[*id as usize] = token.clone();
            }
        }

        let mut special_tokens = HashMap::new();
        for spec in [
            "<pad>", "<bos>", "<eos>", "<unk>",
            "<prichs_think>", "</prichs_think>",
            "<starkdb_query>", "</starkdb_query>", "<stark_action>"
        ] {
            if let Some(&id) = vocab.get(spec) {
                special_tokens.insert(spec.to_string(), id);
            }
        }

        Ok(Self {
            hf_tokenizer: Some(tok),
            vocab,
            id_to_token,
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        if let Some(ref tok) = self.hf_tokenizer {
            tok.get_vocab_size(true)
        } else {
            self.id_to_token.len()
        }
    }

    pub fn add_special_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.special_tokens.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        self.special_tokens.insert(token.to_string(), id);
        id
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref tok) = self.hf_tokenizer {
            match tok.encode(text, false) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(_) => text.as_bytes().iter().map(|&b| b as u32).collect(),
            }
        } else {
            text.as_bytes().iter().map(|&b| b as u32).collect()
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(ref tok) = self.hf_tokenizer {
            match tok.decode(ids, false) {
                Ok(text) => text,
                Err(_) => {
                    let bytes: Vec<u8> = ids.iter().map(|&id| (id & 0xFF) as u8).collect();
                    String::from_utf8_lossy(&bytes).to_string()
                }
            }
        } else {
            let bytes: Vec<u8> = ids.iter().map(|&id| (id & 0xFF) as u8).collect();
            String::from_utf8_lossy(&bytes).to_string()
        }
    }
}

impl Default for PrichsTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_v2_special_tokens_and_roundtrip() {
        let tok_path = Path::new("../../tokenizer.json");
        let tok_path = if tok_path.exists() {
            tok_path
        } else {
            Path::new("tokenizer.json")
        };
        if !tok_path.exists() {
            return; // Skip if file not found in test context
        }

        let tokenizer = PrichsTokenizer::from_huggingface_json(tok_path)
            .expect("Failed to load tokenizer v2");

        assert_eq!(tokenizer.vocab_size(), 32768);

        // Check special token IDs
        let think_id = *tokenizer.special_tokens.get("<prichs_think>").expect("think token missing");
        let think_close_id = *tokenizer.special_tokens.get("</prichs_think>").expect("think close missing");
        let eos_id = *tokenizer.special_tokens.get("<eos>").expect("eos missing");

        assert_eq!(think_id, 4);
        assert_eq!(think_close_id, 5);
        assert_eq!(eos_id, 2);

        // Check text round-trip
        let phrase = "<prichs_think>Аналізую архітектуру StarkOS та MoE.</prichs_think>Всі системи онлайн.";
        let ids = tokenizer.encode(phrase);
        assert_eq!(ids[0], think_id);
        
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, phrase);
    }
}
