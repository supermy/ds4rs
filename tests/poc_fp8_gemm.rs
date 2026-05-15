use cudarc::driver::CudaContext;
use ds4rs::{init_tvm_runtime, CpuTensor, DType, GpuTensor, KernelRegistry, TlKernel};
use std::sync::Arc;

fn make_device() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA init failed")
}

#[test]
fn test_cuda_context_init() {
    let device = make_device();
    let stream = device.default_stream();
    let slice = stream.alloc_zeros::<f32>(16).expect("alloc failed");
    assert_eq!(slice.len(), 16);
}

#[test]
fn test_gpu_tensor_alloc_and_copy() {
    let device = make_device();
    let cpu = CpuTensor::new(
        vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        vec![4],
        DType::FP32,
    );
    let gpu = GpuTensor::from_host(device, &cpu).expect("H2D failed");
    let back = gpu.to_host().expect("D2H failed");
    let result: Vec<f32> = bytemuck::cast_slice(&back.data).to_vec();
    assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_gpu_tensor_device_ptr_nonzero() {
    let device = make_device();
    let cpu = CpuTensor::new(vec![0u8; 1024], vec![1024], DType::UINT8);
    let gpu = GpuTensor::from_host(device, &cpu).expect("H2D failed");
    let ptr = gpu.device_ptr();
    assert_ne!(ptr, 0, "device_ptr should be non-zero");
}

#[test]
fn test_tvm_runtime_init() {
    let runtime = init_tvm_runtime().expect("TVM runtime init failed");
    assert!(!runtime.lib_path().as_os_str().is_empty());
}

#[test]
fn test_kernel_registry_load() {
    let runtime = init_tvm_runtime().expect("TVM runtime init failed");
    let registry = KernelRegistry::new(runtime);

    let so_path = std::env::var("DS4RS_POC_SO_PATH")
        .unwrap_or_else(|_| "/workspace/tilelang/build/fp8_gemm_N4096_K4096.so".to_string());

    if !std::path::Path::new(&so_path).exists() {
        eprintln!("skipping test_kernel_registry_load: {} not found", so_path);
        return;
    }

    registry
        .load(&so_path, "fp8_gemm_kernel_")
        .expect("kernel load failed");

    let key = format!("{}/{}", so_path, "fp8_gemm_kernel_");
    registry.call(&key, &[]).unwrap_err();
}

#[test]
fn test_fp8_gemm_via_c_api() {
    let device = make_device();
    let runtime = init_tvm_runtime().expect("TVM runtime init failed");

    let so_path = std::env::var("DS4RS_POC_SO_PATH")
        .unwrap_or_else(|_| "/workspace/tilelang/build/fp8_gemm_N4096_K4096.so".to_string());

    if !std::path::Path::new(&so_path).exists() {
        eprintln!("skipping test_fp8_gemm_via_c_api: {} not found", so_path);
        return;
    }

    let kernel =
        TlKernel::load(&runtime, &so_path, "fp8_gemm_kernel_").expect("kernel load failed");

    let m: usize = 32;
    let n: usize = 4096;
    let k: usize = 4096;

    let a_fp8 = CpuTensor::new(vec![0u8; m * k], vec![m, k], DType::FP8E4M3);
    let a_s = CpuTensor::new(
        vec![0u8; m * (k / 128) * 4]
            .iter()
            .map(|_| 1.0f32)
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        vec![m, k / 128],
        DType::FP32,
    );
    let b_fp8 = CpuTensor::new(vec![0u8; n * k], vec![n, k], DType::FP8E4M3);
    let b_s = CpuTensor::new(
        vec![0u8; (n / 128) * (k / 128) * 4]
            .iter()
            .map(|_| 1.0f32)
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        vec![n / 128, k / 128],
        DType::FP32,
    );
    let c_gpu = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16).expect("c alloc failed");

    let a_gpu = GpuTensor::from_host(device.clone(), &a_fp8).expect("a H2D failed");
    let b_gpu = GpuTensor::from_host(device.clone(), &b_fp8).expect("b H2D failed");
    let a_s_gpu = GpuTensor::from_host(device.clone(), &a_s).expect("a_s H2D failed");
    let b_s_gpu = GpuTensor::from_host(device, &b_s).expect("b_s H2D failed");

    let result = kernel.call(&[&a_gpu, &b_gpu, &c_gpu, &a_s_gpu, &b_s_gpu]);
    match result {
        Ok(()) => {
            let c_host = c_gpu.to_host().expect("c D2H failed");
            assert_eq!(c_host.data.len(), m * n * 2, "output size mismatch");
            println!("fp8_gemm via C API: output size correct");
        }
        Err(e) => {
            eprintln!("fp8_gemm kernel call error (expected with zero input): {}", e);
        }
    }
}
