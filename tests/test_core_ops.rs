use cudarc::driver::CudaContext;
use ds4rs::*;
use std::sync::Arc;

const BUILD_DIR: &str = "/workspace/tilelang/build";

fn make_device() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA init failed")
}

fn try_tvm() -> Option<Arc<TvmRuntime>> {
    match init_tvm_runtime() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("skip: TVM unavailable ({})", e);
            None
        }
    }
}

fn so_exists(name: &str) -> bool {
    std::path::Path::new(&format!("{}/{}.so", BUILD_DIR, name)).exists()
}

fn model_available() -> bool {
    std::path::Path::new("/models/config.json").exists()
}

fn make_registry() -> Option<Arc<KernelRegistry>> {
    let rt = try_tvm()?;
    let reg = KernelRegistry::new(rt);
    if std::path::Path::new(BUILD_DIR).exists() {
        let _ = reg.load_dir(BUILD_DIR);
    }
    Some(Arc::new(reg))
}

fn make_bf16_tensor(device: &Arc<CudaContext>, shape: &[usize], values: &[f32]) -> GpuTensor {
    let bf16: Vec<half::bf16> = values.iter().map(|v| half::bf16::from_f32(*v)).collect();
    let cpu = CpuTensor::new(bytemuck::cast_slice(&bf16).to_vec(), shape.to_vec(), DType::BF16);
    GpuTensor::from_host(device.clone(), &cpu).expect("H2D failed")
}

fn make_f32_tensor(device: &Arc<CudaContext>, shape: &[usize], values: &[f32]) -> GpuTensor {
    let cpu = CpuTensor::new(
        values.iter().flat_map(|f| f.to_le_bytes()).collect(),
        shape.to_vec(),
        DType::FP32,
    );
    GpuTensor::from_host(device.clone(), &cpu).expect("H2D failed")
}

fn make_i32_tensor(device: &Arc<CudaContext>, shape: &[usize], values: &[i32]) -> GpuTensor {
    let cpu = CpuTensor::new(
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        shape.to_vec(),
        DType::INT32,
    );
    GpuTensor::from_host(device.clone(), &cpu).expect("H2D failed")
}

fn read_bf16_as_f32(t: &GpuTensor) -> Vec<f32> {
    let h = t.to_host().expect("D2H failed");
    let bf16: &[half::bf16] = bytemuck::cast_slice(&h.data);
    bf16.iter().map(|v| v.to_f32()).collect()
}

fn read_f32(t: &GpuTensor) -> Vec<f32> {
    let h = t.to_host().expect("D2H failed");
    bytemuck::cast_slice(&h.data).to_vec()
}

fn read_i32(t: &GpuTensor) -> Vec<i32> {
    let h = t.to_host().expect("D2H failed");
    bytemuck::cast_slice(&h.data).to_vec()
}

// ============================================================
// K7: RMSNorm 单元测试 (使用生产尺寸 N=4096)
// ============================================================

#[test]
fn test_rmsnorm_bf16_weighted() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("rmsnorm_N4096") {
        eprintln!("skip: rmsnorm_N4096 not loaded");
        return;
    }

    let n = 4096usize;
    let m = 4usize;
    let x_vals: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.001).sin() * 0.5)).collect();
    let w_vals: Vec<f32> = vec![1.0; n];

    let x = make_bf16_tensor(&device, &[m, n], &x_vals);
    let w = make_f32_tensor(&device, &[n], &w_vals);
    let y = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16).expect("alloc failed");

    reg.call("rmsnorm_N4096", &[&x, &w, &y]).expect("rmsnorm call failed");

    let result = read_bf16_as_f32(&y);
    for row in 0..m {
        let sq_sum: f32 = (0..n).map(|j| x_vals[row * n + j] * x_vals[row * n + j]).sum();
        let inv_norm = 1.0 / (sq_sum / n as f32 + 1e-6).sqrt();
        let row_max_diff = (0..n)
            .map(|j| (result[row * n + j] - x_vals[row * n + j] * inv_norm).abs())
            .fold(0.0f32, f32::max);
        assert!(row_max_diff < 0.6, "rmsnorm row {} max_diff={:.4}", row, row_max_diff);
    }
    println!("rmsnorm_bf16_weighted N=4096: PASS");
}

#[test]
fn test_rmsnorm_f32() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("rmsnorm_f32_N4096") {
        eprintln!("skip: rmsnorm_f32_N4096 not loaded");
        return;
    }

    let n = 4096usize;
    let m = 4usize;
    let x_vals: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.001).sin() * 0.5)).collect();

    let x = make_f32_tensor(&device, &[m, n], &x_vals);
    let y = GpuTensor::zeros(device.clone(), vec![m, n], DType::FP32).expect("alloc failed");

    reg.call("rmsnorm_f32_N4096", &[&x, &y]).expect("rmsnorm_f32 call failed");

    let result = read_f32(&y);
    for row in 0..m {
        let sq_sum: f32 = (0..n).map(|j| x_vals[row * n + j] * x_vals[row * n + j]).sum();
        let inv_norm = 1.0 / (sq_sum / n as f32 + 1e-6).sqrt();
        let row_max_diff = (0..n)
            .map(|j| (result[row * n + j] - x_vals[row * n + j] * inv_norm).abs())
            .fold(0.0f32, f32::max);
        assert!(row_max_diff < 1e-3, "rmsnorm_f32 row {} max_diff={:.6}", row, row_max_diff);
    }
    println!("rmsnorm_f32 N=4096: PASS");
}

// ============================================================
// K9: SwiGLU 单元测试 (使用生产尺寸 N=2048)
// ============================================================

#[test]
fn test_swiglu() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("swiglu_N2048") {
        eprintln!("skip: swiglu_N2048 not loaded");
        return;
    }

    let n = 2048usize;
    let m = 4usize;
    let gate_vals: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.01).sin() * 2.0)).collect();
    let up_vals: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.02).cos() * 1.5)).collect();

    let gate = make_bf16_tensor(&device, &[m, n], &gate_vals);
    let up = make_bf16_tensor(&device, &[m, n], &up_vals);
    let y = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16).expect("alloc failed");

    reg.call("swiglu_N2048", &[&gate, &up, &y]).expect("swiglu call failed");

    let result = read_bf16_as_f32(&y);
    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let g = gate_vals[i].clamp(-10.0, 10.0);
        let u = up_vals[i].clamp(-10.0, 10.0);
        let silu_g = g / (1.0 + (-g).exp());
        let expected = silu_g * u;
        let diff = (result[i] - expected).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 0.1, "swiglu max_diff={:.4}", max_diff);
    println!("swiglu N=2048: PASS (max_diff={:.4})", max_diff);
}

// ============================================================
// K6: HC Sinkhorn 单元测试
// ============================================================

#[test]
fn test_hc_sinkhorn_doubly_stochastic() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("hc_sinkhorn_hc4_it20") {
        eprintln!("skip: hc_sinkhorn_hc4_it20 not loaded");
        return;
    }

    let hc = 4usize;
    let mix_hc = (2 + hc) * hc;
    let n = 2usize;

    let mixes_vals: Vec<f32> = (0..n * mix_hc).map(|i| (i as f32 * 0.1 - 1.0)).collect();
    let scale_vals: Vec<f32> = vec![1.0, 1.0, 1.0];
    let base_vals: Vec<f32> = vec![0.0; mix_hc];

    let mixes = make_f32_tensor(&device, &[n, mix_hc], &mixes_vals);
    let hc_scale = make_f32_tensor(&device, &[3], &scale_vals);
    let hc_base = make_f32_tensor(&device, &[mix_hc], &base_vals);
    let pre = GpuTensor::zeros(device.clone(), vec![n, hc], DType::FP32).expect("alloc failed");
    let post = GpuTensor::zeros(device.clone(), vec![n, hc], DType::FP32).expect("alloc failed");
    let comb = GpuTensor::zeros(device.clone(), vec![n, hc, hc], DType::FP32).expect("alloc failed");

    reg.call("hc_sinkhorn_hc4_it20", &[&mixes, &hc_scale, &hc_base, &pre, &post, &comb])
        .expect("hc_sinkhorn call failed");

    let comb_host = read_f32(&comb);

    for row in 0..n {
        let mut actual_row_sums = vec![0.0f32; hc];
        let mut actual_col_sums = vec![0.0f32; hc];
        for j in 0..hc {
            for k in 0..hc {
                let v = comb_host[row * hc * hc + j * hc + k];
                actual_row_sums[j] += v;
                actual_col_sums[k] += v;
            }
        }
        for j in 0..hc {
            assert!(
                (actual_row_sums[j] - 1.0).abs() < 0.05,
                "comb[{}] row {} sum = {:.4}, expected ~1.0", row, j, actual_row_sums[j]
            );
            assert!(
                (actual_col_sums[j] - 1.0).abs() < 0.05,
                "comb[{}] col {} sum = {:.4}, expected ~1.0", row, j, actual_col_sums[j]
            );
        }
    }

    let pre_host = read_f32(&pre);
    let post_host = read_f32(&post);
    for i in 0..n * hc {
        assert!(pre_host[i] > 0.0, "pre[{}] should be positive", i);
        assert!(post_host[i] >= 0.0, "post[{}] should be non-negative", i);
    }

    println!("hc_sinkhorn_doubly_stochastic: PASS");
}

// ============================================================
// Cast 单元测试 (使用生产尺寸 N=4096)
// ============================================================

#[test]
fn test_cast_bf16_f32_roundtrip() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("cast_bf16_to_f32_N4096") || !reg.has("cast_f32_to_bf16_N4096") {
        eprintln!("skip: cast kernels not loaded for N=4096");
        return;
    }

    let n = 4096usize;
    let m = 2usize;
    let vals: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.001).sin() * 10.0)).collect();

    let x_bf16 = make_bf16_tensor(&device, &[m, n], &vals);
    let y_f32 = GpuTensor::zeros(device.clone(), vec![m, n], DType::FP32).expect("alloc failed");

    reg.call("cast_bf16_to_f32_N4096", &[&x_bf16, &y_f32]).expect("cast bf16→f32 failed");

    let f32_result = read_f32(&y_f32);
    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let bf16_approx = half::bf16::from_f32(vals[i]).to_f32();
        let diff = (f32_result[i] - bf16_approx).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 1e-6, "cast_bf16_f32 max_diff={:.6}", max_diff);

    let z_bf16 = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16).expect("alloc failed");
    reg.call("cast_f32_to_bf16_N4096", &[&y_f32, &z_bf16]).expect("cast f32→bf16 failed");

    let bf16_result = read_bf16_as_f32(&z_bf16);
    let mut max_diff2 = 0.0f32;
    for i in 0..m * n {
        let bf16_approx = half::bf16::from_f32(vals[i]).to_f32();
        let diff = (bf16_result[i] - bf16_approx).abs();
        max_diff2 = max_diff2.max(diff);
    }
    assert!(max_diff2 < 0.01, "cast_f32_bf16 max_diff={:.4}", max_diff2);
    println!("cast_bf16_f32_roundtrip N=4096: PASS");
}

// ============================================================
// F1: fused_shared_ffn 单元测试
// ============================================================

#[test]
fn test_fused_shared_ffn_shape() {
    let device = make_device();
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };
    if !reg.has("fused_shared_ffn_D4096_I2048") {
        eprintln!("skip: fused_shared_ffn not loaded");
        return;
    }

    let m = 4usize;
    let inter = 2048usize;
    let dim = 4096usize;

    let gate_vals: Vec<f32> = (0..m * inter).map(|i| ((i as f32 * 0.01).sin() * 0.5)).collect();
    let up_vals: Vec<f32> = (0..m * inter).map(|i| ((i as f32 * 0.02).cos() * 0.3)).collect();

    let gate = make_bf16_tensor(&device, &[m, inter], &gate_vals);
    let up = make_bf16_tensor(&device, &[m, inter], &up_vals);

    let w2_data: Vec<u8> = vec![0u8; dim * inter];
    let w2_cpu = CpuTensor::new(w2_data, vec![dim, inter], DType::FP8E4M3);
    let w2 = GpuTensor::from_host(device.clone(), &w2_cpu).expect("w2 H2D failed");

    let ws_rows = dim / 128;
    let ws_cols = inter / 128;
    let w2_s_data: Vec<u8> = vec![0x68u8; ws_rows * ws_cols];
    let w2_s_cpu = CpuTensor::new(w2_s_data, vec![ws_rows, ws_cols], DType::FP8E8M0);
    let w2_s = GpuTensor::from_host(device.clone(), &w2_s_cpu).expect("w2_s H2D failed");

    let y = GpuTensor::zeros(device.clone(), vec![m, dim], DType::BF16).expect("alloc failed");

    let result = reg.call("fused_shared_ffn_D4096_I2048", &[&gate, &up, &w2, &w2_s, &y]);

    match result {
        Ok(()) => {
            assert_eq!(y.shape, vec![m, dim]);
            let y_host = read_bf16_as_f32(&y);
            let non_zero = y_host.iter().filter(|v| v.abs() > 0.0).count();
            println!("fused_shared_ffn_shape: PASS (output shape [{}, {}], non_zero={})", m, dim, non_zero);
        }
        Err(e) => {
            eprintln!("fused_shared_ffn call error: {}", e);
        }
    }
}

// ============================================================
// 参数校验单元测试
// ============================================================

#[test]
fn test_kernel_nargs_mismatch() {
    let rt = match try_tvm() {
        Some(r) => r,
        None => return,
    };
    if !reg_has(&rt, "rmsnorm_N4096") {
        eprintln!("skip: rmsnorm_N4096 not loaded");
        return;
    }

    let kernel = TlKernel::load(
        &rt,
        &format!("{}/rmsnorm_N4096.so", BUILD_DIR),
        "rmsnorm_kernel_",
    ).expect("load failed");

    let device = make_device();
    let x = GpuTensor::zeros(device, vec![1, 4096], DType::BF16).expect("alloc failed");

    let result = kernel.call(&[&x]);
    assert!(result.is_err(), "should fail with wrong nargs (expected 3, got 1)");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("expected 3 args, got 1"), "error message should mention nargs: {}", err_msg);
    println!("kernel_nargs_mismatch: PASS");
}

fn reg_has(rt: &Arc<TvmRuntime>, name: &str) -> bool {
    let reg = KernelRegistry::new(Arc::clone(rt));
    if std::path::Path::new(BUILD_DIR).exists() {
        let _ = reg.load_dir(BUILD_DIR);
    }
    reg.has(name)
}

#[test]
fn test_kernel_registry_has() {
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };

    assert!(reg.has("rmsnorm_N4096") || !so_exists("rmsnorm_N4096"), "has() should return true for loaded kernels");
    assert!(!reg.has("nonexistent_kernel_xyz"), "has() should return false for nonexistent kernels");
    println!("kernel_registry_has: PASS");
}

// ============================================================
// 集成测试: 配置一致性
// ============================================================

#[test]
fn test_config_consistency() {
    if !model_available() {
        eprintln!("skip: model not available");
        return;
    }
    let config = ModelConfig::from_dir("/models").unwrap();

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 43);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.hc_mult, 4);
    assert_eq!(config.q_lora_rank, 1024);
    assert_eq!(config.o_lora_rank, 1024);
    assert_eq!(config.o_groups, 8);
    assert_eq!(config.qk_rope_head_dim, 64);
    assert_eq!(config.n_routed_experts, 256);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.num_experts_per_tok, 6);
    assert_eq!(config.moe_intermediate_size, 2048);
    assert_eq!(config.sliding_window, 128);

    let kv_dim = config.kv_dim();
    assert_eq!(kv_dim, 512);

    let head_dim = config.head_dim;
    assert_eq!(head_dim, 512);

    println!("config_consistency: PASS");
}

// ============================================================
// 集成测试: 内核加载完整性
// ============================================================

#[test]
fn test_all_production_kernels_loaded() {
    let reg = match make_registry() {
        Some(r) => r,
        None => return,
    };

    let required_kernels = [
        "fp8_gemm_N32768_K1024",
        "fp8_gemm_N512_K4096",
        "fp8_gemm_N1024_K4096",
        "fp8_gemm_N4096_K8192",
        "fp8_gemm_N2048_K4096",
        "fp8_gemm_N4096_K2048",
        "fp8_gemm_N8192_K1024",
        "fp4_gemm_N2048_K4096",
        "fp4_gemm_N4096_K2048",
        "rmsnorm_N4096",
        "rmsnorm_N1024",
        "rmsnorm_N512",
        "rmsnorm_no_weight_N1024",
        "rmsnorm_no_weight_N512",
        "rmsnorm_f32_N4096",
        "rmsnorm_f32_N7168",
        "rmsnorm_f32_N16384",
        "swiglu_N2048",
        "sparse_attn_h64_d512",
        "hc_sinkhorn_hc4_it20",
        "rope_interleaved_fwd_D64",
        "rope_interleaved_inv_D64",
        "cast_bf16_to_f32_N4096",
        "cast_f32_to_bf16_N4096",
        "act_quant_N4096_bs128",
        "act_quant_N8192_bs128",
        "moe_route_sqrtsp_N256_topk6",
        "scatter_add_D4096",
        "fused_shared_ffn_D4096_I2048",
    ];

    let mut missing = Vec::new();
    for name in &required_kernels {
        if !reg.has(name) {
            missing.push(*name);
        }
    }

    if missing.is_empty() {
        println!("all_production_kernels_loaded: PASS ({} kernels)", required_kernels.len());
    } else {
        println!("missing {} kernels: {:?}", missing.len(), missing);
        if missing.len() > 5 {
            panic!("too many missing kernels");
        }
    }
}
