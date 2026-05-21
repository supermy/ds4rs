use anyhow::{anyhow, Result};
use std::path::Path;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
}

impl Tokenizer {
    pub fn from_dir(model_dir: &str) -> Result<Self> {
        let tokenizer_path = Path::new(model_dir).join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        let bos_id = inner
            .token_to_id("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>")
            .unwrap_or(0);
        let eos_id = inner
            .token_to_id("<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>")
            .ok_or_else(|| anyhow!("eos token not found in tokenizer"))?;

        Ok(Self { inner, bos_id, eos_id })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {}", e))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if add_bos && !ids.is_empty() && ids[0] != self.bos_id {
            ids.insert(0, self.bos_id);
        }
        if ids.len() >= 2 && ids[0] == self.bos_id && ids[1] == self.bos_id {
            ids.remove(0);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self
            .inner
            .decode(ids, true)
            .map_err(|e| anyhow!("tokenizer decode failed: {}", e))?;
        Ok(text)
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// 参照官方 encoding_dsv4.py 的 encode_messages
/// chat 模式下，<｜Assistant｜> 后必须跟 thinking_end_token (token 128822)
/// thinking_end_token = "</think" + ">" = 8 chars
/// thinking_start_token = "<think" + ">" = 7 chars
pub fn encode_chat(messages: &[ChatMessage], thinking_mode: bool) -> String {
    let bos = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
    let user_sp = "<\u{ff5c}User\u{ff5c}>";
    let assistant_sp = "<\u{ff5c}Assistant\u{ff5c}>";
    let eos = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
    // 官方 encoding_dsv4.py: thinking_end_token = "</think" (U+003C U+002F U+0074 U+0068 U+0069 U+006E U+006B U+003E)
    let thinking_end_token = "\u{3c}\u{2f}\u{74}\u{68}\u{69}\u{6e}\u{6b}\u{3e}";
    // 官方 encoding_dsv4.py: thinking_start_token = "<think" (U+003C U+0074 U+0068 U+0069 U+006E U+006B U+003E)
    let thinking_start_token = "\u{3c}\u{74}\u{68}\u{69}\u{6e}\u{6b}\u{3e}";

    let mut parts = Vec::new();
    parts.push(bos.to_string());

    for msg in messages {
        match msg.role {
            Role::System => {
                parts.push(msg.content.clone());
            }
            Role::User => {
                parts.push(format!("{}{}", user_sp, msg.content));
            }
            Role::Assistant => {
                if thinking_mode {
                    parts.push(format!("{}{}{}{}{}", assistant_sp, thinking_start_token, msg.content, thinking_end_token, eos));
                } else {
                    parts.push(format!("{}{}{}{}", assistant_sp, thinking_end_token, msg.content, eos));
                }
            }
        }
    }

    // 官方 encoding_dsv4.py: chat 模式下 user 后追加 <｜Assistant｜> + thinking_end_token
    parts.push(format!("{}{}", assistant_sp, thinking_end_token));
    parts.join("")
}

#[derive(Debug, Clone)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
}

pub fn parse_assistant_response(text: &str) -> String {
    let eos = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
    text.split(eos).next().unwrap_or(text).to_string()
}
