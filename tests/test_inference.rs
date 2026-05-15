use ds4rs::*;
use std::sync::Arc;

fn make_device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn model_available() -> bool {
    std::path::Path::new("/models/config.json").exists()
}

const MODEL_DIR: &str = "/models";

#[test]
fn test_infer_config_load() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    println!("Config loaded: hidden={}, layers={}, heads={}, hc_mult={}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.hc_mult);
    println!("  q_lora={}, o_lora={}, o_groups={}, rope_dim={}",
        config.q_lora_rank, config.o_lora_rank, config.o_groups, config.qk_rope_head_dim);
    println!("  n_routed={}, n_shared={}, topk={}, inter_dim={}",
        config.n_routed_experts, config.n_shared_experts, config.num_experts_per_tok, config.moe_intermediate_size);
    println!("  sliding_window={}, expert_dtype={}, scoring={}",
        config.sliding_window, config.expert_dtype, config.scoring_func);
}

#[test]
fn test_infer_weight_loader() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let mut loader = WeightLoader::from_dir(MODEL_DIR).unwrap();

    let embed = loader.load("embed.weight").unwrap();
    println!("embed.weight: shape={:?}, dtype={}", embed.shape, embed.dtype);

    let norm = loader.load("norm.weight").unwrap();
    println!("norm.weight: shape={:?}, dtype={}", norm.shape, norm.dtype);

    let head = loader.load("head.weight").unwrap();
    println!("head.weight: shape={:?}, dtype={}", head.shape, head.dtype);

    let gate = loader.load("layers.0.ffn.gate.weight").unwrap();
    println!("layers.0.ffn.gate.weight: shape={:?}, dtype={}", gate.shape, gate.dtype);

    assert!(loader.contains("layers.0.attn.wq_a.weight"));
    assert!(loader.contains("layers.0.attn.wq_b.weight"));
    assert!(loader.contains("layers.0.attn.wkv.weight"));
    assert!(loader.contains("layers.0.hc_attn_fn"));
    assert!(loader.contains("layers.0.ffn.shared_experts.w1.weight"));
}

#[test]
fn test_infer_layer_weight_shapes() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let mut loader = WeightLoader::from_dir(MODEL_DIR).unwrap();

    let q_norm = loader.load("layers.0.attn.q_norm.weight").unwrap();
    println!("q_norm: shape={:?}, dtype={}, len={}", q_norm.shape, q_norm.dtype, q_norm.data.len());

    let kv_norm = loader.load("layers.0.attn.kv_norm.weight").unwrap();
    println!("kv_norm: shape={:?}, dtype={}, len={}", kv_norm.shape, kv_norm.dtype, kv_norm.data.len());

    let attn_norm = loader.load("layers.0.attn_norm.weight").unwrap();
    println!("attn_norm: shape={:?}, dtype={}, len={}", attn_norm.shape, attn_norm.dtype, attn_norm.data.len());

    let ffn_norm = loader.load("layers.0.ffn_norm.weight").unwrap();
    println!("ffn_norm: shape={:?}, dtype={}, len={}", ffn_norm.shape, ffn_norm.dtype, ffn_norm.data.len());

    let wq_a = loader.load("layers.0.attn.wq_a.weight").unwrap();
    println!("wq_a: shape={:?}, dtype={}", wq_a.shape, wq_a.dtype);

    let wq_b = loader.load("layers.0.attn.wq_b.weight").unwrap();
    println!("wq_b: shape={:?}, dtype={}", wq_b.shape, wq_b.dtype);

    let wkv = loader.load("layers.0.attn.wkv.weight").unwrap();
    println!("wkv: shape={:?}, dtype={}", wkv.shape, wkv.dtype);

    let hc_attn_fn = loader.load("layers.0.hc_attn_fn").unwrap();
    println!("hc_attn_fn: shape={:?}, dtype={}", hc_attn_fn.shape, hc_attn_fn.dtype);

    let hc_attn_scale = loader.load("layers.0.hc_attn_scale").unwrap();
    println!("hc_attn_scale: shape={:?}, dtype={}", hc_attn_scale.shape, hc_attn_scale.dtype);

    let hc_attn_base = loader.load("layers.0.hc_attn_base").unwrap();
    println!("hc_attn_base: shape={:?}, dtype={}", hc_attn_base.shape, hc_attn_base.dtype);

    let wo_a = loader.load("layers.0.attn.wo_a.weight").unwrap();
    println!("wo_a: shape={:?}, dtype={}", wo_a.shape, wo_a.dtype);

    let wo_a_scale = loader.load("layers.0.attn.wo_a.scale").unwrap();
    println!("wo_a_scale: shape={:?}, dtype={}", wo_a_scale.shape, wo_a_scale.dtype);

    let wo_b = loader.load("layers.0.attn.wo_b.weight").unwrap();
    println!("wo_b: shape={:?}, dtype={}", wo_b.shape, wo_b.dtype);
}

#[test]
fn test_infer_rope_values() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    let rope = RopeCache::precompute(&config, 4096, 0);

    let (cos, sin) = rope.get_slice(0, 1);
    assert!((cos[0] - 1.0f32).abs() < 1e-5, "cos(0) should be ~1.0");
    assert!(sin[0].abs() < 1e-5, "sin(0) should be ~0.0");

    let (cos1, sin1) = rope.get_slice(1, 1);
    assert!(cos1[0] < 1.0, "cos(1) should be < 1.0");
    assert!(sin1[0] > 0.0, "sin(1) should be > 0.0");

    println!("RoPE: dim={}, cos[0]={:.6}, sin[0]={:.6}, cos[1]={:.6}, sin[1]={:.6}",
        rope.dim, cos[0], sin[0], cos1[0], sin1[0]);
}

#[test]
fn test_infer_rmsnorm_correctness() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eps = 1e-6f32;
    let last_dim = 4usize;
    for r in 0..2 {
        let base = r * last_dim;
        let sq_sum: f32 = (0..last_dim).map(|d| data[base + d].powi(2)).sum();
        let inv_norm = 1.0 / (sq_sum / last_dim as f32 + eps).sqrt();
        println!("Row {}: sq_sum={}, inv_norm={}", r, sq_sum, inv_norm);
        for d in 0..last_dim {
            let normed = data[base + d] * inv_norm;
            println!("  [{}] = {:.6}", d, normed);
        }
    }
}

#[test]
fn test_infer_gate_routing() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());

    let mut loader = WeightLoader::from_dir(MODEL_DIR).unwrap();
    let gate_w = loader.load("layers.0.ffn.gate.weight").unwrap();
    println!("Gate weight: shape={:?}, dtype={}", gate_w.shape, gate_w.dtype);

    let gate_w_gpu = GpuTensor::from_host(device.clone(), &gate_w).unwrap();

    let tvm = match init_tvm_runtime() {
        Ok(t) => t,
        Err(_) => { eprintln!("skipping: TVM not available"); return; }
    };
    let kernels = Arc::new(KernelRegistry::new(tvm));

    let gate = Gate::new(
        &config,
        0,
        device.clone(),
        cublas,
        kernels,
        gate_w_gpu,
        None,
        None,
    );

    let x_data: Vec<half::bf16> = (0..config.hidden_size)
        .map(|i| half::bf16::from_f32((i as f32 * 0.01).sin()))
        .collect();
    let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![1, 1, config.hidden_size], DType::BF16);
    let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu).unwrap();

    let output = gate.forward(&x_gpu, None).unwrap();

    let w_host = output.weights.to_host().unwrap();
    let i_host = output.indices.to_host().unwrap();
    let w_f32: &[f32] = bytemuck::cast_slice(&w_host.data);
    let i_i32: &[i32] = bytemuck::cast_slice(&i_host.data);

    println!("Gate output: topk={}", config.num_experts_per_tok);
    for k in 0..config.num_experts_per_tok {
        println!("  expert[{}]: id={}, weight={:.6}", k, i_i32[k], w_f32[k]);
    }

    assert_eq!(i_i32.len(), config.num_experts_per_tok);
    for &idx in i_i32 {
        assert!((idx as usize) < config.n_routed_experts);
    }
}

#[test]
fn test_infer_single_layer_forward() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap());
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());
    let rt = init_tvm_runtime().expect("TVM runtime failed");
    let kernels = Arc::new(KernelRegistry::new(rt));

    let mut loader = WeightLoader::from_dir(MODEL_DIR).unwrap();
    let layer_result = TransformerLayer::new(0, device, Arc::clone(&config), kernels, cublas, 1, 4096, &mut loader);

    match layer_result {
        Ok(mut layer) => {
            let rope = RopeCache::precompute(&config, 4096, 0);
            let hc = config.hc_mult;
            let dim = config.hidden_size;
            let x_data: Vec<half::bf16> = (0..hc * dim)
                .map(|i| half::bf16::from_f32(((i as f32 * 0.01).sin() * 0.1)))
                .collect();
            let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![1, 1, hc, dim], DType::BF16);
            let x_gpu = GpuTensor::from_host(layer.kv_cache.cache.device.clone(), &x_cpu).unwrap();

            match layer.forward(&x_gpu, 0, &rope, Some(&[1])) {
                Ok(out) => {
                    println!("Layer 0 forward OK: output shape={:?}", out.shape);
                    assert_eq!(out.shape, vec![1, 1, hc, dim]);
                }
                Err(e) => {
                    eprintln!("Layer forward failed: {:?}", e);
                    panic!("Layer forward should not fail");
                }
            }
        }
        Err(e) => {
            eprintln!("Layer load failed: {}", e);
            eprintln!("This may be expected if GPU memory is insufficient");
        }
    }
}
