use ds4rs::*;
use std::sync::Arc;

fn make_device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn make_kernels() -> Option<Arc<KernelRegistry>> {
    let tvm = init_tvm_runtime().ok()?;
    Some(Arc::new(KernelRegistry::new(tvm)))
}

fn model_available() -> bool {
    std::path::Path::new("/models/config.json").exists()
}

const MODEL_DIR: &str = "/models";

#[test]
fn test_gate_score_func_from_str() {
    assert_eq!(ScoreFunc::from_str("softmax"), ScoreFunc::Softmax);
    assert_eq!(ScoreFunc::from_str("sigmoid"), ScoreFunc::Sigmoid);
    assert_eq!(ScoreFunc::from_str("sqrtsoftplus"), ScoreFunc::SqrtSoftplus);
    assert_eq!(ScoreFunc::from_str("unknown"), ScoreFunc::SqrtSoftplus);
}

#[test]
fn test_gate_route_scores_softmax() {
    let device = make_device();
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());
    let kernels = match make_kernels() {
        Some(k) => k,
        None => { eprintln!("skipping: TVM not available"); return; }
    };

    let n_experts = 8;
    let topk = 2;
    let dim = 4;
    let bsz = 1;
    let seqlen = 1;

    let gate_w_data: Vec<half::bf16> = (0..n_experts * dim)
        .map(|i| half::bf16::from_f32(i as f32 * 0.1))
        .collect();
    let gate_w_cpu = CpuTensor::new(
        bytemuck::cast_slice(&gate_w_data).to_vec(),
        vec![n_experts, dim],
        DType::BF16,
    );
    let gate_w_gpu = GpuTensor::from_host(device.clone(), &gate_w_cpu).unwrap();

    let mut config = ModelConfig::default();
    config.n_routed_experts = n_experts;
    config.num_experts_per_tok = topk;
    config.scoring_func = "softmax".to_string();
    config.routed_scaling_factor = 1.0;

    let gate = Gate::new(&config, 0, device.clone(), cublas, kernels, gate_w_gpu, None, None);

    let x_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0].iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect();
    let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![bsz, seqlen, dim], DType::BF16);
    let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu).unwrap();

    let result = gate.forward(&x_gpu, None);
    assert!(result.is_ok(), "Gate forward failed: {:?}", result.err());

    let output = result.unwrap();
    assert_eq!(output.weights.shape, vec![bsz, seqlen, topk]);
    assert_eq!(output.indices.shape, vec![bsz, seqlen, topk]);

    let w_host = output.weights.to_host().unwrap();
    let i_host = output.indices.to_host().unwrap();
    let w_f32: &[f32] = bytemuck::cast_slice(&w_host.data);
    let i_i32: &[i32] = bytemuck::cast_slice(&i_host.data);

    for &idx in i_i32 {
        assert!((idx as usize) < n_experts, "expert index {} out of range", idx);
    }

    let weight_sum: f32 = w_f32.iter().sum();
    assert!((weight_sum - 1.0).abs() < 0.01, "softmax weights should sum to ~1.0, got {}", weight_sum);

    println!("Gate: indices={:?}, weights={:?}", i_i32, w_f32);
}

#[test]
fn test_gate_route_scores_sqrtsoftplus() {
    let device = make_device();
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());
    let kernels = match make_kernels() {
        Some(k) => k,
        None => { eprintln!("skipping: TVM not available"); return; }
    };

    let n_experts = 8;
    let topk = 2;
    let dim = 4;

    let gate_w_data: Vec<half::bf16> = (0..n_experts * dim)
        .map(|i| half::bf16::from_f32(i as f32 * 0.1))
        .collect();
    let gate_w_cpu = CpuTensor::new(
        bytemuck::cast_slice(&gate_w_data).to_vec(),
        vec![n_experts, dim],
        DType::BF16,
    );
    let gate_w_gpu = GpuTensor::from_host(device.clone(), &gate_w_cpu).unwrap();

    let mut config = ModelConfig::default();
    config.n_routed_experts = n_experts;
    config.num_experts_per_tok = topk;
    config.scoring_func = "sqrtsoftplus".to_string();
    config.routed_scaling_factor = 1.0;

    let gate = Gate::new(&config, 0, device.clone(), cublas, kernels, gate_w_gpu, None, None);

    let x_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0, 4.0].iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect();
    let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![1, 1, dim], DType::BF16);
    let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu).unwrap();

    let output = gate.forward(&x_gpu, None).unwrap();

    let w_host = output.weights.to_host().unwrap();
    let w_f32: &[f32] = bytemuck::cast_slice(&w_host.data);

    let weight_sum: f32 = w_f32.iter().sum();
    assert!((weight_sum - 1.0).abs() < 0.01, "sqrtsoftplus weights should sum to ~1.0, got {}", weight_sum);

    println!("SqrtSoftplus Gate: weights={:?}", w_f32);
}

#[test]
fn test_expert_scheduler_create() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap());
    let scheduler = ExpertScheduler::new(device, config.clone());

    assert_eq!(scheduler.expert_count(), 0);
    assert_eq!(scheduler.inter_dim, config.moe_intermediate_size);
    assert_eq!(scheduler.dim, config.hidden_size);
    println!("ExpertScheduler: inter_dim={}, dim={}, expert_dtype={:?}",
        scheduler.inter_dim, scheduler.dim, scheduler.expert_dtype);
}

#[test]
fn test_fp4_dequant_basic() {
    let packed_data: Vec<u8> = vec![0x12, 0x34];
    let scales: Vec<u8> = vec![0x80, 0x80];
    let packed_shape = vec![1, 2];
    let logical_k = 4;

    let result = dequant_fp4_e2m1_to_bf16(&packed_data, &scales, &packed_shape, logical_k);
    assert!(result.is_ok());

    let out = result.unwrap();
    assert_eq!(out.shape, vec![1, logical_k]);

    let out_f32: Vec<f32> = bytemuck::cast_slice::<u8, half::bf16>(&out.data)
        .iter().map(|x| x.to_f32()).collect();
    for val in &out_f32 {
        assert!(val.is_finite(), "FP4 dequant produced non-finite: {}", val);
    }
    println!("FP4 dequant: {:?}", out_f32);
}

#[test]
fn test_fp8_dequant_basic() {
    let data: Vec<u8> = vec![0x00, 0x40, 0x80, 0xC0];
    let scales: Vec<u8> = vec![0x80; 4];
    let shape = vec![1, 4];

    let result = dequant_fp8_e4m3_to_bf16(&data, &scales, &shape);
    assert!(result.is_ok());

    let out = result.unwrap();
    assert_eq!(out.shape, vec![1, 4]);

    let out_f32: Vec<f32> = bytemuck::cast_slice::<u8, half::bf16>(&out.data)
        .iter().map(|x| x.to_f32()).collect();
    for val in &out_f32 {
        assert!(val.is_finite(), "FP8 dequant produced non-finite: {}", val);
    }
    println!("FP8 dequant: {:?}", out_f32);
}

#[test]
fn test_gate_with_bias() {
    let device = make_device();
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());
    let kernels = match make_kernels() {
        Some(k) => k,
        None => { eprintln!("skipping: TVM not available"); return; }
    };

    let n_experts = 4;
    let topk = 2;
    let dim = 4;

    let gate_w_data: Vec<half::bf16> = (0..n_experts * dim)
        .map(|i| half::bf16::from_f32(i as f32 * 0.1))
        .collect();
    let gate_w_cpu = CpuTensor::new(
        bytemuck::cast_slice(&gate_w_data).to_vec(),
        vec![n_experts, dim],
        DType::BF16,
    );
    let gate_w_gpu = GpuTensor::from_host(device.clone(), &gate_w_cpu).unwrap();

    let bias_data: Vec<half::bf16> = vec![10.0, -10.0, 5.0, -5.0].iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect();
    let bias_cpu = CpuTensor::new(bytemuck::cast_slice(&bias_data).to_vec(), vec![n_experts], DType::BF16);
    let bias_gpu = GpuTensor::from_host(device.clone(), &bias_cpu).unwrap();

    let mut config = ModelConfig::default();
    config.n_routed_experts = n_experts;
    config.num_experts_per_tok = topk;
    config.scoring_func = "sigmoid".to_string();
    config.routed_scaling_factor = 1.0;

    let gate = Gate::new(&config, 0, device.clone(), cublas, kernels, gate_w_gpu, Some(bias_gpu), None);

    let x_data: Vec<half::bf16> = vec![1.0, 0.0, 0.0, 0.0].iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect();
    let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![1, 1, dim], DType::BF16);
    let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu).unwrap();

    let output = gate.forward(&x_gpu, None).unwrap();

    let i_host = output.indices.to_host().unwrap();
    let i_i32: &[i32] = bytemuck::cast_slice(&i_host.data);

    assert_eq!(i_i32.len(), topk);
    for &idx in i_i32 {
        assert!((idx as usize) < n_experts);
    }

    println!("Gate with bias: indices={:?}", i_i32);
}
