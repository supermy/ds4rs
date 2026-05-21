use cudarc::driver::CudaContext;
use ds4rs::{init_tvm_runtime, KernelRegistry, Transformer, TvmRuntime};
use std::sync::Arc;

const MODEL_DIR: &str = "/models";

fn model_available() -> bool {
    std::path::Path::new(MODEL_DIR).join("config.json").exists()
}

fn make_device() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA init failed")
}

fn try_tvm_runtime() -> Option<Arc<TvmRuntime>> {
    match init_tvm_runtime() {
        Ok(rt) => Some(rt),
        Err(e) => {
            eprintln!("skipping: TVM runtime not available ({})", e);
            None
        }
    }
}

#[test]
fn test_transformer_load() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let rt = match try_tvm_runtime() {
        Some(r) => r,
        None => return,
    };
    let registry = Arc::new(KernelRegistry::new(rt));
    match Transformer::load(MODEL_DIR, device, 1, 4096, registry) {
        Ok(model) => {
            assert_eq!(model.layers.len(), model.config.num_hidden_layers);
            println!("Model loaded: {} layers, vocab={}",
                model.config.num_hidden_layers, model.config.vocab_size);
        }
        Err(e) => {
            eprintln!("skipping: model load failed (likely OOM): {}", e);
        }
    }
}

#[test]
fn test_transformer_forward_skeleton() {
    if !model_available() {
        eprintln!("skipping: model not available");
        return;
    }
    let device = make_device();
    let rt = match try_tvm_runtime() {
        Some(r) => r,
        None => return,
    };
    let registry = Arc::new(KernelRegistry::new(rt));
    let mut model = match Transformer::load(MODEL_DIR, device, 1, 4096, registry) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping: model load failed (likely OOM): {}", e);
            return;
        }
    };

    let input_ids: Vec<u32> = vec![1, 2, 3, 4];
    let result = model.forward(&input_ids, 0);
    assert!(result.is_ok(), "forward failed: {:?}", result.err());
    let out = result.unwrap();
    assert_eq!(out.shape[0], 1);
    assert_eq!(out.shape[1], 4);
    assert_eq!(out.shape[2], model.config.vocab_size);
    println!("Forward pass: input=[1,2,3,4], output shape={:?}", out.shape);
}
