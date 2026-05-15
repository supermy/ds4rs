use ds4rs::{init_tvm_runtime, KernelRegistry, TlKernel, TvmRuntime};
use std::sync::Arc;

const BUILD_DIR: &str = "/workspace/tilelang/build";

fn runtime() -> Arc<TvmRuntime> {
    init_tvm_runtime().expect("TVM runtime init failed")
}

fn so_exists(name: &str) -> bool {
    std::path::Path::new(&format!("{}/{}.so", BUILD_DIR, name)).exists()
}

#[test]
fn test_load_act_quant_kernel() {
    let rt = runtime();
    if !so_exists("act_quant_N4096_bs128") {
        eprintln!("skipping: kernel .so not found");
        return;
    }
    let kernel = TlKernel::load(
        &rt,
        &format!("{}/act_quant_N4096_bs128.so", BUILD_DIR),
        "act_quant_kernel_",
    );
    assert!(kernel.is_ok(), "act_quant load failed: {:?}", kernel.err());
}

#[test]
fn test_load_fp8_gemm_kernel() {
    let rt = runtime();
    if !so_exists("fp8_gemm_N512_K4096") {
        eprintln!("skipping: kernel .so not found");
        return;
    }
    let kernel = TlKernel::load(
        &rt,
        &format!("{}/fp8_gemm_N512_K4096.so", BUILD_DIR),
        "fp8_gemm_kernel_",
    );
    assert!(kernel.is_ok(), "fp8_gemm load failed: {:?}", kernel.err());
}

#[test]
fn test_load_rmsnorm_kernel() {
    let rt = runtime();
    if !so_exists("rmsnorm_N1024") {
        eprintln!("skipping: kernel .so not found");
        return;
    }
    let kernel = TlKernel::load(
        &rt,
        &format!("{}/rmsnorm_N1024.so", BUILD_DIR),
        "rmsnorm_kernel_",
    );
    assert!(kernel.is_ok(), "rmsnorm load failed: {:?}", kernel.err());
}

#[test]
fn test_load_swiglu_kernel() {
    let rt = runtime();
    if !so_exists("swiglu_N2048") {
        eprintln!("skipping: kernel .so not found");
        return;
    }
    let kernel = TlKernel::load(
        &rt,
        &format!("{}/swiglu_N2048.so", BUILD_DIR),
        "swiglu_kernel_",
    );
    assert!(kernel.is_ok(), "swiglu load failed: {:?}", kernel.err());
}

#[test]
fn test_load_hc_sinkhorn_kernel() {
    let rt = runtime();
    if !so_exists("hc_sinkhorn_hc4_it20") {
        eprintln!("skipping: kernel .so not found");
        return;
    }
    let kernel = TlKernel::load(
        &rt,
        &format!("{}/hc_sinkhorn_hc4_it20.so", BUILD_DIR),
        "hc_split_sinkhorn_kernel_",
    );
    assert!(kernel.is_ok(), "hc_sinkhorn load failed: {:?}", kernel.err());
}

#[test]
fn test_kernel_registry_load_dir() {
    let rt = runtime();
    if !std::path::Path::new(BUILD_DIR).exists() {
        eprintln!("skipping: build dir not found");
        return;
    }
    let registry = KernelRegistry::new(rt);
    let count = registry.load_dir(BUILD_DIR).expect("load_dir failed");
    println!("Loaded {} kernels from {}", count, BUILD_DIR);
    assert!(count > 0, "no kernels loaded");
}
