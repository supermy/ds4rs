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
fn test_gpu_expert_cache_create() {
    let device = make_device();
    let config = Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap_or_else(|_| {
        ModelConfig::from_file("nonexistent").unwrap()
    }));
    let cache = GpuExpertCache::new(device, config, 4);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_capacity(), 4);
}

#[test]
fn test_gpu_expert_cache_put_get() {
    let device = make_device();
    let config = if model_available() {
        Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap())
    } else {
        return eprintln!("skipping: model not available");
    };

    let mut cache = GpuExpertCache::new(device.clone(), config, 4);

    let w1 = GpuTensor::zeros(device.clone(), vec![2048, 4096], DType::BF16).unwrap();
    let w1_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
    let w3 = GpuTensor::zeros(device.clone(), vec![2048, 4096], DType::BF16).unwrap();
    let w3_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
    let w2 = GpuTensor::zeros(device.clone(), vec![4096, 2048], DType::BF16).unwrap();
    let w2_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();

    let weights = ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale };

    cache.put(0, 0, weights).unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(0, 0));

    let result = cache.get(0, 0);
    assert!(result.is_some());
    println!("GPU cache: get(0,0) OK, shape={:?}", result.unwrap().w1.shape);
}

#[test]
fn test_gpu_expert_cache_lfu_eviction() {
    let device = make_device();
    let config = if model_available() {
        Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap())
    } else {
        return eprintln!("skipping: model not available");
    };

    let mut cache = GpuExpertCache::new(device.clone(), config, 2);

    for expert_id in 0..2 {
        let w1 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
        let w1_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
        let w3 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
        let w3_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
        let w2 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
        let w2_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();

        let weights = ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale };
        cache.put(0, expert_id, weights).unwrap();
    }

    let _ = cache.get(0, 0);
    let _ = cache.get(0, 0);
    let _ = cache.get(0, 1);

    let w1 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
    let w1_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
    let w3 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
    let w3_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();
    let w2 = GpuTensor::zeros(device.clone(), vec![128, 128], DType::BF16).unwrap();
    let w2_scale = GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).unwrap();

    let weights = ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale };
    cache.put(0, 2, weights).unwrap();

    assert_eq!(cache.len(), 2);
    assert!(cache.contains(0, 0));
    assert!(!cache.contains(0, 1));
    assert!(cache.contains(0, 2));
    println!("LFU eviction: evicted expert 1 (least freq), remaining experts 0,2");
}

#[test]
fn test_ram_expert_cache_slru() {
    let mut cache = RamExpertCache::new(1, 2);

    let data1 = vec![1u8; 512 * 1024];
    let data2 = vec![2u8; 512 * 1024];
    let data3 = vec![3u8; 512 * 1024];

    cache.put(0, 0, data1.clone());
    cache.put(0, 1, data2.clone());
    cache.put(0, 2, data3.clone());

    assert!(cache.contains(0, 0));
    assert!(cache.contains(0, 1));
    assert!(cache.contains(0, 2));

    let result = cache.get(0, 2);
    assert!(result.is_some());
    assert_eq!(result.unwrap()[0], 3u8);

    println!("RAM SLRU cache: hot={}, cold={}", cache.hot_len(), cache.cold_len());
}

#[test]
fn test_ram_cache_promotion() {
    let mut cache = RamExpertCache::new(1, 2);

    let data1 = vec![1u8; 512 * 1024];
    let data2 = vec![2u8; 512 * 1024];
    let data3 = vec![3u8; 512 * 1024];

    cache.put(0, 0, data1);
    cache.put(0, 1, data2);
    cache.put(0, 2, data3);

    let result = cache.get(0, 2);
    assert!(result.is_some());

    println!("RAM cache promotion: hot={}, cold={}", cache.hot_len(), cache.cold_len());
}

#[test]
fn test_ssd_expert_cache() {
    let tmp_dir = std::env::temp_dir().join("ds4rs_test_ssd_cache");
    let _ = std::fs::create_dir_all(&tmp_dir);

    let mut cache = SsdExpertCache::new(tmp_dir.to_str().unwrap());

    let data = vec![42u8; 1024];
    cache.put_indexed(0, 5, &data).unwrap();

    let result = cache.get(0, 5);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), data);

    let mmap_result = cache.mmap_prefetch(0, 5);
    assert!(mmap_result.is_some());
    assert_eq!(&mmap_result.unwrap()[..10], &[42u8; 10]);

    let _ = std::fs::remove_dir_all(&tmp_dir);
    println!("SSD cache: put/get/mmap OK");
}

#[test]
fn test_ssd_cache_scan_index() {
    let tmp_dir = std::env::temp_dir().join("ds4rs_test_ssd_scan");
    let _ = std::fs::create_dir_all(tmp_dir.join("experts"));

    let data = vec![1u8; 100];
    std::fs::write(tmp_dir.join("experts/0_0"), &data).unwrap();
    std::fs::write(tmp_dir.join("experts/1_3"), &data).unwrap();

    let mut cache = SsdExpertCache::new(tmp_dir.to_str().unwrap());
    let count = cache.scan_index();

    assert_eq!(count, 2);
    assert!(cache.contains(0, 0));
    assert!(cache.contains(1, 3));

    let _ = std::fs::remove_dir_all(&tmp_dir);
    println!("SSD cache scan: found {} experts", count);
}

#[test]
fn test_three_level_cache_create() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap());
    let tmp_dir = std::env::temp_dir().join("ds4rs_test_3level");

    let cache = ThreeLevelCache::new(
        device,
        config,
        8,
        512,
        2048,
        tmp_dir.to_str().unwrap(),
    );

    assert_eq!(cache.gpu.max_capacity(), 8);
    assert_eq!(cache.gpu.len(), 0);

    let _ = std::fs::remove_dir_all(&tmp_dir);
    println!("ThreeLevelCache created: gpu_slots=8, ram_hot=512MB, ram_cold=2048MB");
}

#[test]
fn test_layer_prefetcher() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let config = Arc::new(ModelConfig::from_dir(MODEL_DIR).unwrap());
    let cublas = Arc::new(CublasHandle::new(device.clone()).unwrap());

    let gate_w_data: Vec<half::bf16> = (0..256 * 4096)
        .map(|i| half::bf16::from_f32(i as f32 * 0.001))
        .collect();
    let gate_w_cpu = CpuTensor::new(
        bytemuck::cast_slice(&gate_w_data).to_vec(),
        vec![256, 4096],
        DType::BF16,
    );
    let gate_w_gpu = GpuTensor::from_host(device.clone(), &gate_w_cpu).unwrap();

    let tvm = match init_tvm_runtime() {
        Ok(t) => t,
        Err(_) => { eprintln!("skipping: TVM not available"); return; }
    };
    let kernels = Arc::new(KernelRegistry::new(tvm));

    let mut gate_config = ModelConfig::default();
    gate_config.n_routed_experts = 256;
    gate_config.num_experts_per_tok = 8;
    gate_config.scoring_func = "sqrtsoftplus".to_string();
    gate_config.routed_scaling_factor = 1.0;

    let gate = Gate::new(&gate_config, 0, device.clone(), cublas, kernels, gate_w_gpu, None, None);

    let x_data: Vec<half::bf16> = (0..4096).map(|i| half::bf16::from_f32(i as f32 * 0.01)).collect();
    let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_data).to_vec(), vec![1, 1, 4096], DType::BF16);
    let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu).unwrap();

    let gate_output = gate.forward(&x_gpu, None).unwrap();

    let prefetcher = LayerPrefetcher::new(config, 1);
    let predicted = prefetcher.predict_next_experts(0, &gate_output);

    assert!(!predicted.is_empty(), "prefetcher should predict at least one expert");
    assert!(predicted.len() <= 8, "predicted experts should not exceed topk");
    for &expert_id in &predicted {
        assert!(expert_id < 256, "expert_id {} out of range", expert_id);
    }

    println!("Prefetcher: predicted {} experts: {:?}", predicted.len(), predicted);
}
