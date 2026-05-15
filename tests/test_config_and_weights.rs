use ds4rs::{DType, ModelConfig, WeightLoader};

const MODEL_DIR: &str = "/models";

fn model_available() -> bool {
    std::path::Path::new(MODEL_DIR).join("config.json").exists()
}

#[test]
fn test_config_from_dir() {
    if !model_available() {
        eprintln!("skipping: model not available at {}", MODEL_DIR);
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).expect("config load failed");
    assert_eq!(config.num_hidden_layers, 43);
    assert_eq!(config.n_routed_experts, 256);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.num_experts_per_tok, 6);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.head_dim, 512);
    assert_eq!(config.vocab_size, 129280);
    println!("Config: {} layers, {} experts, hidden={}",
        config.num_hidden_layers, config.n_routed_experts, config.hidden_size);
}

#[test]
fn test_config_computed_fields() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).expect("config load failed");
    assert_eq!(config.kv_dim(), config.num_key_value_heads * config.head_dim);
    assert_eq!(config.q_dim(), config.num_attention_heads * config.head_dim);
    assert!(config.head_dim_non_rope() > 0);
    assert!(config.total_layers() >= config.num_hidden_layers);
    println!("kv_dim={}, q_dim={}, head_dim_non_rope={}",
        config.kv_dim(), config.q_dim(), config.head_dim_non_rope());
}

#[test]
fn test_weight_loader_index() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let loader = WeightLoader::from_dir(MODEL_DIR).expect("weight loader init failed");
    assert!(loader.contains("embed.weight"));
    assert!(loader.contains("layers.0.attn.wq_a.weight"));
    assert!(loader.contains("layers.0.ffn.experts.0.w1.weight"));
    assert!(loader.contains("layers.0.ffn.shared_experts.w1.weight"));
    println!("Total weight keys: {}", loader.list_keys().len());
}

#[test]
fn test_weight_loader_load_single() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let mut loader = WeightLoader::from_dir(MODEL_DIR).expect("weight loader init failed");
    let tensor = loader.load("embed.weight").expect("load embed.weight failed");
    assert!(!tensor.data.is_empty());
    assert_eq!(tensor.dtype, DType::BF16);
    assert!(tensor.shape.len() >= 2);
    println!("embed.weight: shape={:?}, dtype={}", tensor.shape, tensor.dtype);
}

#[test]
fn test_weight_loader_load_expert() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let mut loader = WeightLoader::from_dir(MODEL_DIR).expect("weight loader init failed");
    let expert = loader.load_expert(0, 0).expect("load expert 0.0 failed");
    assert!(!expert.is_empty());
    let w1_key = String::from("layers.0.ffn.experts.0.w1.weight");
    assert!(expert.contains_key(&w1_key), "missing {}", w1_key);
    let w1 = &expert[&w1_key];
    assert_eq!(w1.dtype, DType::UINT8, "FP4 packed as uint8");
    println!("Expert 0.0: {} tensors, w1 shape={:?}", expert.len(), w1.shape);
}

#[test]
fn test_weight_loader_load_shared_experts() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let mut loader = WeightLoader::from_dir(MODEL_DIR).expect("weight loader init failed");
    let shared = loader.load_shared_experts(0).expect("load shared experts failed");
    assert!(!shared.is_empty());
    println!("Shared experts layer 0: {} tensors", shared.len());
}

#[test]
fn test_weight_loader_expert_keys() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let loader = WeightLoader::from_dir(MODEL_DIR).expect("weight loader init failed");
    let keys = loader.layer_expert_keys(0, 0);
    assert!(!keys.is_empty());
    for key in &keys {
        assert!(key.starts_with("layers.0.ffn.experts.0."));
    }
    println!("Expert 0.0 keys: {:?}", keys);
}
