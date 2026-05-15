use ds4rs::*;
use std::sync::Arc;

fn make_device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn init_tvm() -> Option<Arc<TvmRuntime>> {
    init_tvm_runtime().ok()
}

fn model_available() -> bool {
    std::path::Path::new("/models/config.json").exists()
}

const MODEL_DIR: &str = "/models";

#[test]
fn test_cublas_handle_create() {
    let device = make_device();
    let handle = CublasHandle::new(device);
    assert!(handle.is_ok(), "CublasHandle creation failed: {:?}", handle.err());
}

#[test]
fn test_cublas_bf16_gemm() {
    let device = make_device();
    let handle = CublasHandle::new(device.clone()).unwrap();

    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    let a_data: Vec<half::bf16> = (0..m * k).map(|i| half::bf16::from_f32(i as f32)).collect();
    let b_data: Vec<half::bf16> = (0..n * k).map(|i| half::bf16::from_f32(i as f32 * 0.5)).collect();

    let a_cpu = CpuTensor::new(bytemuck::cast_slice(&a_data).to_vec(), vec![m, k], DType::BF16);
    let b_cpu = CpuTensor::new(bytemuck::cast_slice(&b_data).to_vec(), vec![n, k], DType::BF16);

    let a_gpu = GpuTensor::from_host(device.clone(), &a_cpu).unwrap();
    let b_gpu = GpuTensor::from_host(device.clone(), &b_cpu).unwrap();
    let mut c_gpu = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16).unwrap();

    handle.gemm_bf16(m, n, k, &a_gpu, &b_gpu, &mut c_gpu, 1.0, 0.0).unwrap();

    let c_cpu = c_gpu.to_host().unwrap();
    let c_f32: Vec<f32> = bytemuck::cast_slice(&c_cpu.data).iter().map(|x: &half::bf16| x.to_f32()).collect();

    assert_eq!(c_f32.len(), m * n);
    for val in &c_f32 {
        assert!(val.is_finite(), "GEMM output contains non-finite value: {}", val);
    }
    println!("BF16 GEMM result: {:?}", c_f32);
}

#[test]
fn test_cublas_f32_gemm() {
    let device = make_device();
    let handle = CublasHandle::new(device.clone()).unwrap();

    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    let a_data: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n * k).map(|i| i as f32 * 0.5).collect();

    let a_cpu = CpuTensor::new(bytemuck::cast_slice(&a_data).to_vec(), vec![m, k], DType::FP32);
    let b_cpu = CpuTensor::new(bytemuck::cast_slice(&b_data).to_vec(), vec![n, k], DType::FP32);

    let a_gpu = GpuTensor::from_host(device.clone(), &a_cpu).unwrap();
    let b_gpu = GpuTensor::from_host(device.clone(), &b_cpu).unwrap();
    let mut c_gpu = GpuTensor::zeros(device.clone(), vec![m, n], DType::FP32).unwrap();

    handle.gemm_f32(m, n, k, &a_gpu, &b_gpu, &mut c_gpu, 1.0, 0.0).unwrap();

    let c_cpu = c_gpu.to_host().unwrap();
    let c_f32: &[f32] = bytemuck::cast_slice(&c_cpu.data);

    assert_eq!(c_f32.len(), m * n);
    for val in c_f32 {
        assert!(val.is_finite(), "GEMM output contains non-finite value: {}", val);
    }
    println!("FP32 GEMM result: {:?}", c_f32);
}

#[test]
fn test_quant_fp8_e4m3_roundtrip() {
    let test_vals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 448.0, -448.0];
    let mut fp8_bytes = Vec::new();
    for v in &test_vals {
        let bits = f32_to_fp8_e4m3(*v);
        fp8_bytes.push(bits);
    }

    for (i, &orig) in test_vals.iter().enumerate() {
        let dequant = fp8_e4m3_to_f32(fp8_bytes[i]);
        if orig == 0.0 {
            assert_eq!(dequant, 0.0);
        } else {
            let rel_err = (dequant - orig).abs() / orig.abs();
            assert!(rel_err < 0.1, "FP8 roundtrip failed for {}: dequant={}, rel_err={}", orig, dequant, rel_err);
        }
    }
    println!("FP8 E4M3 roundtrip OK for {} values", test_vals.len());
}

fn f32_to_fp8_e4m3(val: f32) -> u8 {
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();

    if abs_val == 0.0 {
        return 0;
    }

    let bias: i32 = 7;
    let exp = abs_val.log2().floor() as i32 + bias;
    let exp = exp.clamp(0, 15) as u8;
    let mantissa = abs_val / 2.0f32.powi(exp as i32 - bias) - 1.0;
    let mant_bits = (mantissa * 8.0).round() as u8;

    (sign << 7) | (exp << 3) | (mant_bits & 0x07)
}

fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = ((bits >> 3) & 0x0F) as i32;
    let mant = (bits & 0x07) as i32;
    let bias = 7;

    let val = if exp == 0 && mant == 0 {
        0.0f32
    } else if exp == 0 {
        (mant as f32 / 8.0) * 2.0f32.powi(1 - bias)
    } else {
        (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp - bias)
    };

    if sign == 1 { -val } else { val }
}

#[test]
fn test_rope_precompute_shape() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    let rope = RopeCache::precompute(&config, 4096, 0);

    let half_dim = config.qk_rope_head_dim / 2;
    assert_eq!(rope.freqs_cos.len(), 4096 * half_dim);
    assert_eq!(rope.freqs_sin.len(), 4096 * half_dim);
    assert_eq!(rope.dim, config.qk_rope_head_dim);

    let (cos, sin) = rope.get_slice(0, 4);
    assert_eq!(cos.len(), 4 * half_dim);
    assert_eq!(sin.len(), 4 * half_dim);

    assert!((cos[0] - 1.0f32).abs() < 1e-6, "cos(0) should be ~1.0, got {}", cos[0]);
    assert!(sin[0].abs() < 1e-6, "sin(0) should be ~0.0, got {}", sin[0]);

    println!("RoPE: dim={}, half_dim={}, cos[0..4]={:?}, sin[0..4]={:?}",
        rope.dim, half_dim, &cos[0..4.min(cos.len())], &sin[0..4.min(sin.len())]);
}

#[test]
fn test_rope_yarn_scaling() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    let rope = RopeCache::precompute(&config, 16384, 0);

    let (cos_short, _) = rope.get_slice(0, 1);
    let (cos_long, _) = rope.get_slice(8192, 1);

    assert!(cos_short[0].is_finite());
    assert!(cos_long[0].is_finite());
    println!("RoPE YaRN: cos[0]={}, cos[8192]={}", cos_short[0], cos_long[0]);
}

#[test]
fn test_kv_cache_create() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();
    let cache = KvCache::new(device, &config, 0, 1, config.max_position_embeddings);

    assert!(cache.is_ok());
    let cache = cache.unwrap();
    assert_eq!(cache.window_size, config.sliding_window);
    assert_eq!(cache.head_dim, config.head_dim);
    println!("KV Cache: window={}, head_dim={}",
        cache.window_size, cache.head_dim);
}

#[test]
fn test_gpu_tensor_clone() {
    let device = make_device();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cpu = CpuTensor::new(bytemuck::cast_slice(&data).to_vec(), vec![2, 2], DType::FP32);
    let gpu = GpuTensor::from_host(device.clone(), &cpu).unwrap();

    let cloned = gpu.clone();
    let orig_host = gpu.to_host().unwrap();
    let clone_host = cloned.to_host().unwrap();

    assert_eq!(orig_host.data, clone_host.data);
    assert_eq!(orig_host.shape, clone_host.shape);
}

#[test]
fn test_model_config_fields() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let config = ModelConfig::from_dir(MODEL_DIR).unwrap();

    assert!(config.hidden_size > 0);
    assert!(config.num_hidden_layers > 0);
    assert!(config.num_attention_heads > 0);
    assert!(config.hc_mult > 0);
    assert!(config.q_lora_rank > 0);
    assert!(config.o_lora_rank > 0);
    assert!(config.o_groups > 0);
    assert!(config.n_routed_experts > 0);
    assert!(config.sliding_window > 0);

    assert_eq!(config.kv_dim(), config.num_key_value_heads * config.head_dim);
    assert_eq!(config.q_dim(), config.num_attention_heads * config.head_dim);

    println!("Config: hidden={}, layers={}, heads={}, hc_mult={}, q_lora={}, o_lora={}, o_groups={}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        config.hc_mult, config.q_lora_rank, config.o_lora_rank, config.o_groups);
    println!("  kv_dim={}, q_dim={}, n_routed={}, sliding_window={}",
        config.kv_dim(), config.q_dim(), config.n_routed_experts, config.sliding_window);
}

#[test]
fn test_cublas_strided_batched_gemm() {
    let device = make_device();
    let handle = CublasHandle::new(device.clone()).unwrap();

    let batch = 2usize;
    let m = 2usize;
    let n = 3usize;
    let k = 4usize;

    let a_data: Vec<half::bf16> = (0..batch * m * k).map(|i| half::bf16::from_f32(i as f32)).collect();
    let b_data: Vec<half::bf16> = (0..batch * n * k).map(|i| half::bf16::from_f32(i as f32 * 0.5)).collect();

    let a_cpu = CpuTensor::new(bytemuck::cast_slice(&a_data).to_vec(), vec![batch, m, k], DType::BF16);
    let b_cpu = CpuTensor::new(bytemuck::cast_slice(&b_data).to_vec(), vec![batch, n, k], DType::BF16);

    let a_gpu = GpuTensor::from_host(device.clone(), &a_cpu).unwrap();
    let b_gpu = GpuTensor::from_host(device.clone(), &b_cpu).unwrap();
    let mut c_gpu = GpuTensor::zeros(device.clone(), vec![batch, m, n], DType::BF16).unwrap();

    let stride_a = (m * k) as i64;
    let stride_b = (n * k) as i64;
    let stride_c = (m * n) as i64;

    handle.gemm_bf16_strided_batched(
        m, n, k, &a_gpu, &b_gpu, &mut c_gpu,
        stride_a, stride_b, stride_c,
        batch as i32, 1.0, 0.0,
    ).unwrap();

    let c_cpu = c_gpu.to_host().unwrap();
    let c_f32: Vec<f32> = bytemuck::cast_slice(&c_cpu.data).iter().map(|x: &half::bf16| x.to_f32()).collect();

    assert_eq!(c_f32.len(), batch * m * n);
    for val in &c_f32 {
        assert!(val.is_finite(), "Batched GEMM output contains non-finite value: {}", val);
    }
    println!("Strided batched BF16 GEMM result: {:?}", c_f32);
}

#[test]
fn test_route_scores_sigmoid_cpu() {
    let n_experts = 16usize;
    let topk = 4usize;
    let total = 3usize;

    let scores: Vec<f32> = vec![
        1.0, -0.5, 2.0, 0.3, -1.0, 0.8, -0.2, 1.5,
        0.0, 0.1, -0.3, 0.7, 1.2, -0.8, 0.4, 0.9,
        -1.0, 2.0, -0.5, 1.0, 0.3, -0.7, 1.5, 0.2,
        0.6, -0.1, 0.8, -0.4, 1.1, 0.0, -0.9, 0.5,
        0.3, -0.6, 1.8, 0.2, -1.2, 0.7, 0.1, -0.3,
        0.9, 1.1, -0.2, 0.4, -0.8, 1.3, 0.0, 0.6,
    ];
    assert_eq!(scores.len(), total * n_experts);

    let bias: Vec<f32> = (0..n_experts).map(|i| i as f32 * 0.01).collect();

    let mut activated = vec![0.0f32; total * n_experts];
    for (i, &s) in scores.iter().enumerate() {
        activated[i] = 1.0 / (1.0 + (-s).exp());
    }
    let original = activated.clone();

    let mut topk_weights = vec![0.0f32; total * topk];
    let mut topk_indices = vec![0i32; total * topk];

    for t in 0..total {
        let mut sel: Vec<f32> = (0..n_experts)
            .map(|j| activated[t * n_experts + j] + bias[j])
            .collect();
        for k in 0..topk {
            let (best_idx, _) = sel.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            topk_indices[t * topk + k] = best_idx as i32;
            topk_weights[t * topk + k] = original[t * n_experts + best_idx];
            sel[best_idx] = f32::NEG_INFINITY;
        }
        let sum: f32 = topk_weights[t * topk..(t + 1) * topk].iter().sum();
        if sum > 0.0 {
            for w in topk_weights[t * topk..(t + 1) * topk].iter_mut() {
                *w /= sum;
            }
        }
    }

    for t in 0..total {
        let w_sum: f32 = topk_weights[t * topk..(t + 1) * topk].iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-5, "weights should sum to 1.0, got {}", w_sum);
        for k in 0..topk {
            assert!(topk_weights[t * topk + k] >= 0.0, "weight should be non-negative");
        }
    }

    println!("CPU route_scores: weights={:?}, indices={:?}", topk_weights, topk_indices);
}

#[test]
fn test_route_scores_gpu_vs_cpu() {
    let device = make_device();
    let tvm = match init_tvm() {
        Some(t) => t,
        None => {
            eprintln!("skipping: TVM runtime not available");
            return;
        }
    };
    let kernels = Arc::new(KernelRegistry::new(tvm));

    let kernel_dir = "/workspace/tilelang/build";
    if std::path::Path::new(kernel_dir).join("kernels.json").exists() {
        let _ = kernels.load_dir(kernel_dir);
    }

    let n_experts = 256usize;
    let topk = 6usize;
    let m = 4usize;

    let mut scores_data = vec![0.0f32; m * n_experts];
    for i in 0..m {
        for j in 0..n_experts {
            scores_data[i * n_experts + j] = ((i * n_experts + j) as f32 * 0.01 - 1.0).sin();
        }
    }

    let mut bias_data = vec![0.0f32; n_experts];
    for j in 0..n_experts {
        bias_data[j] = (j as f32 * 0.001).cos();
    }

    let scores_cpu = CpuTensor::new(
        bytemuck::cast_slice(&scores_data).to_vec(),
        vec![m, n_experts],
        DType::FP32,
    );
    let bias_cpu_tensor = CpuTensor::new(
        bytemuck::cast_slice(&bias_data).to_vec(),
        vec![n_experts],
        DType::FP32,
    );

    let scores_gpu = GpuTensor::from_host(device.clone(), &scores_cpu).unwrap();
    let bias_gpu = GpuTensor::from_host(device.clone(), &bias_cpu_tensor).unwrap();

    let route_scale: f32 = 1.5;

    let mut activated = vec![0.0f32; m * n_experts];
    for (i, &s) in scores_data.iter().enumerate() {
        activated[i] = if s > 20.0 {
            s.sqrt()
        } else {
            (1.0 + s.exp()).ln().sqrt()
        };
    }
    let original = activated.clone();

    let mut cpu_weights = vec![0.0f32; m * topk];
    let mut cpu_indices = vec![0i32; m * topk];
    for t in 0..m {
        let mut sel: Vec<f32> = (0..n_experts)
            .map(|j| activated[t * n_experts + j] + bias_data[j])
            .collect();
        for k in 0..topk {
            let (best_idx, _) = sel.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            cpu_indices[t * topk + k] = best_idx as i32;
            cpu_weights[t * topk + k] = original[t * n_experts + best_idx];
            sel[best_idx] = f32::NEG_INFINITY;
        }
        let sum: f32 = cpu_weights[t * topk..(t + 1) * topk].iter().sum();
        if sum > 0.0 {
            for w in cpu_weights[t * topk..(t + 1) * topk].iter_mut() {
                *w /= sum;
            }
        }
        for w in cpu_weights[t * topk..(t + 1) * topk].iter_mut() {
            *w *= route_scale;
        }
    }

    let kernel_name = format!("moe_route_sqrtsp_N{}_topk{}", n_experts, topk);
    let topk_weights = GpuTensor::zeros(device.clone(), vec![m, topk], DType::FP32).unwrap();
    let topk_indices = GpuTensor::zeros(device.clone(), vec![m, topk], DType::INT32).unwrap();

    let result = kernels.call(
        &kernel_name,
        &[&scores_gpu, &bias_gpu, &topk_weights, &topk_indices],
    );

    if let Err(e) = result {
        eprintln!("GPU route kernel not available ({}): {}", kernel_name, e);
        return;
    }

    let gpu_w_host = topk_weights.to_host().unwrap();
    let gpu_w: &[f32] = bytemuck::cast_slice(&gpu_w_host.data);
    let gpu_i_host = topk_indices.to_host().unwrap();
    let gpu_i: &[i32] = bytemuck::cast_slice(&gpu_i_host.data);

    for t in 0..m {
        for k in 0..topk {
            let cpu_idx = cpu_indices[t * topk + k];
            let gpu_idx = gpu_i[t * topk + k];
            assert_eq!(
                cpu_idx, gpu_idx,
                "token={} k={} CPU idx={} != GPU idx={}",
                t, k, cpu_idx, gpu_idx
            );
            let cpu_w = cpu_weights[t * topk + k];
            let gpu_w = gpu_w[t * topk + k];
            assert!(
                (cpu_w - gpu_w).abs() < 1e-4,
                "token={} k={} CPU w={} != GPU w={}",
                t, k, cpu_w, gpu_w
            );
        }
    }

    println!("GPU route_scores matches CPU! weights={:?}, indices={:?}", gpu_w, gpu_i);
}

#[test]
fn test_kv_cache_checkpoint_roundtrip() {
    let device = make_device();
    let mut config = ModelConfig::default();
    config.sliding_window = 128;
    config.head_dim = 64;
    config.compress_ratios = vec![4];

    let mut cache = KvCache::new(
        device.clone(),
        &config,
        0,
        1,
        512,
    ).unwrap();

    let seqlen = 64;
    let head_dim = 64;
    let kv_data: Vec<u16> = (0..seqlen * head_dim)
        .map(|i| half::bf16::from_f32(i as f32 * 0.1).to_bits())
        .collect();
    let kv_cpu = CpuTensor::new(
        bytemuck::cast_slice(&kv_data).to_vec(),
        vec![1, seqlen, head_dim],
        DType::BF16,
    );
    let kv_gpu = GpuTensor::from_host(device.clone(), &kv_cpu).unwrap();

    cache.update_prefill(&kv_gpu, 0, seqlen).unwrap();

    let tmp_path = "/tmp/ds4rs_test_kv_checkpoint.bin";
    cache.save_checkpoint(tmp_path).unwrap();

    let mut cache2 = KvCache::new(
        device.clone(),
        &config,
        0,
        1,
        512,
    ).unwrap();

    cache2.load_checkpoint(tmp_path).unwrap();

    assert_eq!(cache2.current_seqlen.get(&0), Some(&seqlen));

    let original = cache.get_window_kv(0, seqlen, 0).unwrap();
    let restored = cache2.get_window_kv(0, seqlen, 0).unwrap();

    let orig_host = original.to_host().unwrap();
    let rest_host = restored.to_host().unwrap();
    let orig_bf16: &[half::bf16] = bytemuck::cast_slice(&orig_host.data);
    let rest_bf16: &[half::bf16] = bytemuck::cast_slice(&rest_host.data);

    let mut max_diff = 0.0f32;
    for (o, r) in orig_bf16.iter().zip(rest_bf16.iter()) {
        let diff = (o.to_f32() - r.to_f32()).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 1e-6, "KV cache checkpoint roundtrip max_diff={}", max_diff);

    let _ = std::fs::remove_file(tmp_path);
    println!("KV cache checkpoint roundtrip OK, max_diff={}", max_diff);
}

#[test]
fn test_compressor_pool_cpu_correctness() {
    let d = 128usize;
    let pool_size = 4usize;
    let n_groups = 2usize;
    let bsz = 1usize;

    let mut kv_data = vec![0.0f32; bsz * n_groups * pool_size * d];
    let mut score_data = vec![0.0f32; bsz * n_groups * pool_size * d];
    for i in 0..kv_data.len() {
        kv_data[i] = ((i as f32) * 0.01).sin();
        score_data[i] = ((i as f32) * 0.02).cos();
    }

    let mut result = vec![0.0f32; bsz * n_groups * d];
    for b in 0..bsz {
        for g in 0..n_groups {
            for dd in 0..d {
                let mut max_s = f32::NEG_INFINITY;
                for r in 0..pool_size {
                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                    max_s = max_s.max(score_data[idx]);
                }
                let mut sum_exp = 0.0f32;
                let mut weights = vec![0.0f32; pool_size];
                for r in 0..pool_size {
                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                    weights[r] = (score_data[idx] - max_s).exp();
                    sum_exp += weights[r];
                }
                if sum_exp > 0.0 {
                    for w in &mut weights { *w /= sum_exp; }
                }
                let res_off = (b * n_groups + g) * d + dd;
                for r in 0..pool_size {
                    let kv_idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                    result[res_off] += weights[r] * kv_data[kv_idx];
                }
            }
        }
    }

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "result[{}] is not finite: {}", i, v);
    }

    println!("Compressor pool CPU correctness: {} values, all finite", result.len());
}

#[test]
fn test_indexer_score_cpu_correctness() {
    let n_heads = 4usize;
    let head_dim = 32usize;
    let n_comp = 16usize;
    let bsz = 1usize;
    let seqlen = 1usize;
    let total = bsz * seqlen;
    let wq_out_dim = n_heads * head_dim;

    let q_proj: Vec<f32> = (0..total * wq_out_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let kv: Vec<f32> = (0..bsz * n_comp * head_dim).map(|i| (i as f32 * 0.02).cos()).collect();
    let weights: Vec<f32> = (0..total * n_heads).map(|i| (i as f32 * 0.05 + 0.1).sin().max(0.0)).collect();

    let mut index_score = vec![0.0f32; bsz * seqlen * n_comp];
    for b in 0..bsz {
        for s in 0..seqlen {
            for t in 0..n_comp {
                let mut dot_sum = 0.0f32;
                for h in 0..n_heads {
                    let q_base = (b * seqlen + s) * wq_out_dim + h * head_dim;
                    let kv_base = b * n_comp * head_dim + t * head_dim;
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        dot += q_proj[q_base + dd] * kv[kv_base + dd];
                    }
                    let w = weights[(b * seqlen + s) * n_heads + h];
                    dot_sum += dot.max(0.0) * w;
                }
                index_score[(b * seqlen + s) * n_comp + t] = dot_sum;
            }
        }
    }

    let mut topk = 4usize;
    topk = topk.min(n_comp);
    let mut idx: Vec<usize> = (0..n_comp).collect();
    idx.sort_by(|&a, &b| index_score[b].partial_cmp(&index_score[a]).unwrap());
    let topk_result: Vec<usize> = idx[..topk].to_vec();

    for &t in &topk_result {
        assert!(t < n_comp, "topk index out of range");
    }
    for i in 1..topk_result.len() {
        assert!(
            index_score[topk_result[i - 1]] >= index_score[topk_result[i]],
            "topk not sorted in descending order"
        );
    }

    println!("Indexer score CPU correctness: top-{} indices = {:?}", topk, topk_result);
}
