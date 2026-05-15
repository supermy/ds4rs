use ds4rs::{ModelConfig, RopeCache};

const MODEL_DIR: &str = "/models";

fn model_available() -> bool {
    std::path::Path::new(MODEL_DIR).join("config.json").exists()
}

fn load_config() -> ModelConfig {
    ModelConfig::from_dir(MODEL_DIR).expect("config load failed")
}

#[test]
fn test_rope_precompute() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = load_config();
    let seqlen = 1024;
    let cache = RopeCache::precompute(&config, seqlen, 0);

    assert_eq!(cache.seqlen, seqlen);
    assert_eq!(cache.dim, config.qk_rope_head_dim);
    assert_eq!(cache.freqs_cos.len(), seqlen * config.qk_rope_head_dim / 2);
    assert_eq!(cache.freqs_sin.len(), seqlen * config.qk_rope_head_dim / 2);

    assert!((cache.freqs_cos[0] - 1.0f32).abs() < 1e-5, "cos(0) should be 1.0");
    assert!(cache.freqs_sin[0].abs() < 1e-5, "sin(0) should be 0.0");

    println!("RoPE cache: seqlen={}, dim={}, cos[0..4]={:?}",
        seqlen, cache.dim, &cache.freqs_cos[0..4.min(cache.freqs_cos.len())]);
}

#[test]
fn test_rope_slice() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = load_config();
    let cache = RopeCache::precompute(&config, 2048, 0);

    let (cos, sin) = cache.get_slice(0, 4);
    assert_eq!(cos.len(), 4 * config.qk_rope_head_dim / 2);
    assert_eq!(sin.len(), 4 * config.qk_rope_head_dim / 2);

    let (cos2, sin2) = cache.get_slice(100, 4);
    assert_eq!(cos2.len(), cos.len());
    assert!(cos2 != cos, "different positions should have different freqs");
}

#[test]
fn test_config_yarn_params() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = load_config();
    assert!(config.beta_fast() > 0);
    assert!(config.beta_slow() > 0);
    assert!(config.rope_factor_for_layer(2) > 1.0);
    assert!(config.original_seq_len_for_layer(2) > 0);
    println!("YaRN: factor={}, orig_seq_len={}, beta_fast={}, beta_slow={}",
        config.rope_factor_for_layer(2),
        config.original_seq_len_for_layer(2),
        config.beta_fast(),
        config.beta_slow());
}
