use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

fn default_one() -> usize { 1 }
fn default_zero_usize() -> usize { 0 }
fn default_empty_string() -> String { String::new() }
fn default_empty_vec() -> Vec<u32> { Vec::new() }

#[derive(Debug, Deserialize)]
pub struct QuantConfig {
    pub activation_scheme: String,
    pub fmt: String,
    pub quant_method: String,
    pub scale_fmt: String,
    pub weight_block_size: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct RopeScalingConfig {
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub rope_type: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_one")]
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub qk_rope_head_dim: usize,
    pub vocab_size: usize,
    pub n_routed_experts: usize,
    #[serde(default = "default_one")]
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub expert_dtype: String,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub sliding_window: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    #[serde(default)]
    pub hc_eps: f64,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    #[serde(default = "default_zero_usize")]
    pub num_hash_layers: usize,
    #[serde(default = "default_zero_usize")]
    pub num_nextn_predict_layers: usize,
    pub swiglu_limit: f32,
    #[serde(default)]
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub compress_rope_theta: f64,
    #[serde(default = "default_empty_vec")]
    pub compress_ratios: Vec<u32>,
    #[serde(default)]
    pub quantization_config: Option<QuantConfig>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default = "default_zero_usize")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_empty_string")]
    pub dtype: String,
    #[serde(default = "default_empty_string")]
    pub model_dir: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 1,
            num_attention_heads: 32,
            num_key_value_heads: 1,
            head_dim: 128,
            q_lora_rank: 512,
            o_lora_rank: 512,
            o_groups: 1,
            qk_rope_head_dim: 64,
            vocab_size: 1,
            n_routed_experts: 8,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            moe_intermediate_size: 1024,
            expert_dtype: "bf16".to_string(),
            routed_scaling_factor: 1.0,
            scoring_func: "sigmoid".to_string(),
            sliding_window: 4096,
            hc_mult: 1,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            index_n_heads: 1,
            index_head_dim: 64,
            index_topk: 1,
            num_hash_layers: 0,
            num_nextn_predict_layers: 0,
            swiglu_limit: 0.0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            compress_rope_theta: 0.0,
            compress_ratios: vec![],
            quantization_config: None,
            rope_scaling: None,
            max_position_embeddings: 4096,
            dtype: "bf16".to_string(),
            model_dir: String::new(),
        }
    }
}

impl ModelConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config from {}", path))?;
        let mut config: ModelConfig = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse config from {}", path))?;
        if config.model_dir.is_empty() {
            if let Some(parent) = Path::new(path).parent() {
                config.model_dir = parent.to_str().unwrap_or(".").to_string();
            }
        }
        Ok(config)
    }

    pub fn from_dir(model_dir: &str) -> Result<Self> {
        let config_path = Path::new(model_dir).join("config.json");
        let mut config = Self::from_file(config_path.to_str().with_context(|| "invalid path encoding")?)?;
        if std::env::var("DS4RS_SWA_ONLY").as_deref() == Ok("1") {
            eprintln!("[config] DS4RS_SWA_ONLY=1: overriding all compress_ratios to 0 (pure sliding window)");
            config.compress_ratios = vec![0; config.num_hidden_layers];
        }
        Ok(config)
    }

    pub fn head_dim_non_rope(&self) -> usize {
        self.head_dim - self.qk_rope_head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn is_hash_layer(&self, layer_id: usize) -> bool {
        layer_id < self.num_hash_layers
    }

    pub fn compress_ratio(&self, layer_id: usize) -> u32 {
        self.compress_ratios.get(layer_id).copied().unwrap_or(0)
    }

    pub fn has_compressor(&self, layer_id: usize) -> bool {
        self.compress_ratio(layer_id) > 0
    }

    pub fn has_indexer(&self, layer_id: usize) -> bool {
        let ratio = self.compress_ratio(layer_id);
        ratio > 0 && ratio <= 4
    }

    pub fn total_layers(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn rope_theta_for_layer(&self, layer_id: usize) -> f64 {
        if self.compress_ratio(layer_id) > 0 && self.compress_rope_theta > 0.0 {
            self.compress_rope_theta
        } else {
            self.rope_theta
        }
    }

    pub fn rope_factor_for_layer(&self, layer_id: usize) -> f64 {
        if self.compress_ratio(layer_id) > 0 {
            self.rope_scaling.as_ref().map(|r| r.factor).unwrap_or(1.0)
        } else {
            1.0
        }
    }

    pub fn original_seq_len_for_layer(&self, layer_id: usize) -> usize {
        if self.compress_ratio(layer_id) > 0 {
            self.rope_scaling.as_ref().map(|r| r.original_max_position_embeddings).unwrap_or(0)
        } else {
            0
        }
    }

    pub fn beta_fast(&self) -> usize {
        self.rope_scaling.as_ref().map(|r| r.beta_fast).unwrap_or(0)
    }

    pub fn beta_slow(&self) -> usize {
        self.rope_scaling.as_ref().map(|r| r.beta_slow).unwrap_or(0)
    }

    pub fn weight_block_size(&self) -> Vec<usize> {
        self.quantization_config.as_ref()
            .map(|q| q.weight_block_size.clone())
            .unwrap_or_else(|| vec![128, 128])
    }
}
