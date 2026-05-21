use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::layer::TransformerLayer;
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use std::sync::Arc;

/// 格式化字节数为人类可读形式
fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// 打印显存使用情况
fn log_vram(device: &Arc<CudaContext>, label: &str) {
    let _ = device; // 仅为确保 device 存活
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let used = total - free;
            eprintln!("[vram] {} 已用 {} / {} (剩余 {})",
                label, fmt_bytes(used), fmt_bytes(total), fmt_bytes(free));
        }
    }
}

/// 打印张量信息：名称、形状、数据类型、大小
fn log_tensor(name: &str, tensor: &CpuTensor, location: &str) {
    eprintln!("[load]   {} {:?} {:?} = {} ({})",
        name, tensor.shape, tensor.dtype, fmt_bytes(tensor.data.len()), location);
}

fn log_tensor_gpu(name: &str, tensor: &GpuTensor) {
    let size = tensor.shape.iter().product::<usize>() * tensor.dtype.element_size();
    eprintln!("[load]   {} {:?} {:?} = {} (GPU)",
        name, tensor.shape, tensor.dtype, fmt_bytes(size));
}

pub struct Transformer {
    pub config: Arc<ModelConfig>,
    pub device: Arc<CudaContext>,
    pub layers: Vec<TransformerLayer>,
    pub embed_cpu: CpuTensor,
    pub head_weight_cpu: CpuTensor,
    pub head_scale: Option<GpuTensor>,
    pub norm_weight: GpuTensor,
    pub hc_head_fn: GpuTensor,
    pub hc_head_scale: GpuTensor,
    pub hc_head_base: GpuTensor,
    pub ropes: Vec<RopeCache>,
    pub kernels: Arc<KernelRegistry>,
    pub cublas: Arc<CublasHandle>,
}

impl Transformer {
    pub fn load(
        model_dir: &str,
        device: Arc<CudaContext>,
        max_batch: usize,
        max_seqlen: usize,
        kernels: Arc<KernelRegistry>,
    ) -> Result<Self> {
        let config = ModelConfig::from_dir(model_dir)?;
        let config = Arc::new(config);
        eprintln!("[load] ─────────────────────────────────────────────────");
        eprintln!("[load] DeepSeek V4 Flash 模型加载");
        eprintln!("[load] 配置: {} 层, hidden={}, heads={}, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.num_attention_heads, config.vocab_size);
        eprintln!("[load] max_batch={}, max_seqlen={}", max_batch, max_seqlen);
        log_vram(&device, "加载前");

        unsafe {
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            let r = cudarc::driver::sys::cuDeviceGetMemPool(&mut pool, 0);
            if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let threshold: usize = 0;
                cudarc::driver::sys::cuMemPoolSetAttribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const usize as *mut std::ffi::c_void,
                );
                eprintln!("[load] CUDA 内存池释放阈值已设为 0 (立即归还释放的显存)");
            }
        }

        let mut loader = WeightLoader::from_dir(model_dir)?;
        let cublas = Arc::new(CublasHandle::new(device.clone())?);

        eprintln!("[load] ── 全局权重 ──");
        let embed_cpu = loader.load("embed.weight").context("embed.weight")?;
        log_tensor("embed.weight", &embed_cpu, "CPU");

        let head_weight_cpu = if loader.contains("head.weight") {
            let head = loader.load("head.weight").context("head.weight")?;
            log_tensor("head.weight (raw)", &head, "CPU");
            let head_f32 = if head.dtype == DType::BF16 {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&head.data);
                let f32_data: Vec<f32> = bf16.iter().map(|v| v.to_f32()).collect();
                CpuTensor::new(bytemuck::cast_slice(&f32_data).to_vec(), head.shape.clone(), DType::FP32)
            } else {
                head
            };
            eprintln!("[load]   head.weight → FP32 {:?} (独立于 embed, 用于 logits 计算)", head_f32.shape);
            head_f32
        } else {
            eprintln!("[load]   WARNING: head.weight not found, falling back to embed.weight BF16");
            let bf16: &[half::bf16] = bytemuck::cast_slice(&embed_cpu.data);
            let f32_data: Vec<f32> = bf16.iter().map(|v| v.to_f32()).collect();
            CpuTensor::new(bytemuck::cast_slice(&f32_data).to_vec(), embed_cpu.shape.clone(), DType::FP32)
        };

        let head_scale = if loader.contains("head.scale") {
            let s = loader.load("head.scale")?;
            let gs = GpuTensor::from_host(device.clone(), &s)?;
            log_tensor_gpu("head.scale", &gs);
            Some(gs)
        } else {
            None
        };

        let norm_cpu = loader.load("norm.weight").context("norm.weight")?;
        log_tensor("norm.weight", &norm_cpu, "CPU→GPU");
        let norm_weight = GpuTensor::from_host(device.clone(), &norm_cpu).context("norm upload")?;

        let hc_head_fn_cpu = loader.load("hc_head_fn").context("hc_head_fn")?;
        log_tensor("hc_head_fn", &hc_head_fn_cpu, "CPU→GPU");
        let hc_head_fn = GpuTensor::from_host(device.clone(), &hc_head_fn_cpu)?;

        let hc_head_scale_cpu = loader.load("hc_head_scale").context("hc_head_scale")?;
        let hc_head_scale = GpuTensor::from_host(device.clone(), &hc_head_scale_cpu)?;

        let hc_head_base_cpu = loader.load("hc_head_base").context("hc_head_base")?;
        let hc_head_base = GpuTensor::from_host(device.clone(), &hc_head_base_cpu)?;

        log_vram(&device, "全局权重后");

        let mut ropes = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            ropes.push(RopeCache::precompute(&config, max_seqlen, layer_id));
        }

        eprintln!("[load] ── Transformer 层 ({}) ──", config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            eprintln!("[load] ── 层 {}/{} (compress_ratio={}) ──",
                layer_id + 1, config.num_hidden_layers, config.compress_ratio(layer_id));
            let layer = TransformerLayer::new(
                layer_id,
                device.clone(),
                Arc::clone(&config),
                Arc::clone(&kernels),
                Arc::clone(&cublas),
                max_batch,
                max_seqlen,
                &mut loader,
            )
            .with_context(|| format!("layer {} load failed", layer_id))?;
            layers.push(layer);

            // 每 10 层或最后一层打印显存状态
            if (layer_id + 1) % 10 == 0 || layer_id + 1 == config.num_hidden_layers {
                log_vram(&device, &format!("层 {}/{}", layer_id + 1, config.num_hidden_layers));
            }
        }

        eprintln!("[load] ─────────────────────────────────────────────────");
        eprintln!("[load] 模型加载完成: {} 层", config.num_hidden_layers);
        log_vram(&device, "加载完成");

        // 打印 RAM 使用估算
        let embed_ram = embed_cpu.data.len();
        let total_ram_estimate = embed_ram + config.num_hidden_layers * 256 * 12 * 1024 * 1024; // 粗略估算专家 CPU 缓存
        eprintln!("[ram]  预估模型占用 {} (embed+专家CPU缓存)",
            fmt_bytes(total_ram_estimate));

        Ok(Self {
            config,
            device,
            layers,
            embed_cpu,
            head_weight_cpu,
            head_scale,
            norm_weight,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            ropes,
            kernels,
            cublas,
        })
    }

    pub fn forward(&mut self, input_ids: &[u32], start_pos: usize) -> Result<GpuTensor> {
        let bsz = 1usize;
        let _seqlen = input_ids.len();
        let _dim = self.config.hidden_size;
        let hc = self.config.hc_mult;

        let h = self.embed_lookup(input_ids, bsz)?;

        {
            static STEP: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let s = STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if s == 0 {
                let h_host = h.to_host()?;
                let h_f32: Vec<f32> = if h_host.dtype == DType::BF16 {
                    let bf16: &[half::bf16] = bytemuck::cast_slice(&h_host.data);
                    bf16.iter().map(|v| v.to_f32()).collect()
                } else {
                    bytemuck::cast_slice(&h_host.data).to_vec()
                };
                let mean = h_f32.iter().sum::<f32>() / h_f32.len() as f32;
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("[debug] embed output: shape={:?} mean={:.6e} min={:.6e} max={:.6e}", h.shape, mean, min, max);
            }
        }

        let h = self.hc_expand(&h, hc)?;

        let mut h = h;
        {
            static STEP: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let s = STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if s == 0 {
                let h_host = h.to_host()?;
                let h_f32: Vec<f32> = if h_host.dtype == DType::BF16 {
                    let bf16: &[half::bf16] = bytemuck::cast_slice(&h_host.data);
                    bf16.iter().map(|v| v.to_f32()).collect()
                } else {
                    bytemuck::cast_slice(&h_host.data).to_vec()
                };
                let mean = h_f32.iter().sum::<f32>() / h_f32.len() as f32;
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("[debug] after hc_expand: shape={:?} mean={:.4} min={:.4} max={:.4}", h.shape, mean, min, max);
            }
        }
        let n_layers = self.layers.len();
        let debug_first_step = {
            static FIRST_STEP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            FIRST_STEP.swap(false, std::sync::atomic::Ordering::Relaxed)
        };
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, start_pos, &self.ropes[layer_id], Some(input_ids))?;
            if debug_first_step && (layer_id < 3 || layer_id % 10 == 0 || layer_id >= n_layers - 5) {
                let h_host = h.to_host()?;
                let h_f32: Vec<f32> = if h_host.dtype == DType::BF16 {
                    let bf16: &[half::bf16] = bytemuck::cast_slice(&h_host.data);
                    bf16.iter().map(|v| v.to_f32()).collect()
                } else {
                    bytemuck::cast_slice(&h_host.data).to_vec()
                };
                let mean = h_f32.iter().sum::<f32>() / h_f32.len() as f32;
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("[debug] after layer {:2}: shape={:?} mean={:.4} min={:.4} max={:.4}", layer_id, h.shape, mean, min, max);
            }
        }

        let h = self.hc_head_reduce(&h)?;

        {
            static COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let c = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if c < 2 {
                let h_host = h.to_host()?;
                let h_f32: Vec<f32> = if h_host.dtype == DType::BF16 {
                    let bf16: &[half::bf16] = bytemuck::cast_slice(&h_host.data);
                    bf16.iter().map(|v| v.to_f32()).collect()
                } else {
                    bytemuck::cast_slice(&h_host.data).to_vec()
                };
                let mean = h_f32.iter().sum::<f32>() / h_f32.len() as f32;
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("[debug] hc_head_reduce #{}: shape={:?} mean={:.4} min={:.4} max={:.4}", c, h.shape, mean, min, max);
            }
        }

        let h = self.rmsnorm(&h, &self.norm_weight)?;

        {
            static COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let c = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if c < 2 {
                let h_host = h.to_host()?;
                let h_f32: Vec<f32> = if h_host.dtype == DType::BF16 {
                    let bf16: &[half::bf16] = bytemuck::cast_slice(&h_host.data);
                    bf16.iter().map(|v| v.to_f32()).collect()
                } else {
                    bytemuck::cast_slice(&h_host.data).to_vec()
                };
                let mean = h_f32.iter().sum::<f32>() / h_f32.len() as f32;
                let max = h_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = h_f32.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!("[debug] after_final_rmsnorm #{}: shape={:?} mean={:.4} min={:.4} max={:.4}", c, h.shape, mean, min, max);
            }
        }

        let logits = self.head_forward(&h)?;

        Ok(logits)
    }

    /// CPU 端 embed gather + H2D：仅传输需要的行，避免将整个 1GB embed 表常驻 GPU
    fn embed_lookup(&self, input_ids: &[u32], bsz: usize) -> Result<GpuTensor> {
        let seqlen = input_ids.len();
        let dim = self.config.hidden_size;
        let elem_size = self.embed_cpu.dtype.element_size();
        let row_bytes = dim * elem_size;

        let mut gathered = vec![0u8; seqlen * row_bytes];
        for (i, &id) in input_ids.iter().enumerate() {
            let src_off = id as usize * row_bytes;
            let dst_off = i * row_bytes;
            gathered[dst_off..dst_off + row_bytes]
                .copy_from_slice(&self.embed_cpu.data[src_off..src_off + row_bytes]);
        }

        let cpu = CpuTensor::new(gathered, vec![bsz, seqlen, dim], self.embed_cpu.dtype);
        GpuTensor::from_host(self.device.clone(), &cpu)
    }

    fn hc_expand(&self, x: &GpuTensor, hc: usize) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = x.shape[2];
        let device = x.device.clone();

        // CPU 端 expand：将 [b,s,d] 复制 hc 份为 [b,s,hc,d]
        let x_host = x.to_host()?;
        let elem_size = x_host.dtype.element_size();
        let row_bytes = dim * elem_size;
        let total = bsz * seqlen;
        let mut expanded = vec![0u8; total * hc * row_bytes];
        for t in 0..total {
            let src_off = t * row_bytes;
            for h in 0..hc {
                let dst_off = t * hc * row_bytes + h * row_bytes;
                expanded[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&x_host.data[src_off..src_off + row_bytes]);
            }
        }
        let cpu = CpuTensor::new(expanded, vec![bsz, seqlen, hc, dim], x_host.dtype);
        GpuTensor::from_host(device, &cpu)
    }

    fn hc_head_reduce(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let hc = x.shape[2];
        let dim = x.shape[3];
        let device = x.device.clone();
        let norm_eps = 1e-6f32;
        let hc_eps = 1e-6f32;

        let x_host = x.to_host()?;
        let x_f32: Vec<f32> = if x_host.dtype == DType::BF16 {
            let bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
            bf16.iter().map(|v| v.to_f32()).collect()
        } else {
            bytemuck::cast_slice(&x_host.data).to_vec()
        };

        let fn_host = self.hc_head_fn.to_host()?;
        let fn_f32: &[f32] = bytemuck::cast_slice(&fn_host.data);
        let scale_host = self.hc_head_scale.to_host()?;
        let scale_f32: &[f32] = bytemuck::cast_slice(&scale_host.data);
        let base_host = self.hc_head_base.to_host()?;
        let base_f32: &[f32] = bytemuck::cast_slice(&base_host.data);

        let n = bsz * seqlen;
        let x_dim = hc * dim;

        let mut mixes = vec![0.0f32; n * hc];
        for i in 0..n {
            let x_row = &x_f32[i * x_dim..(i + 1) * x_dim];
            let mean_sq = x_row.iter().map(|v| v * v).sum::<f32>() / x_dim as f32;
            let rsqrt = 1.0 / (mean_sq + norm_eps).sqrt();

            for h in 0..hc {
                let mut dot = 0.0f32;
                for d in 0..x_dim {
                    dot += x_row[d] * fn_f32[h * x_dim + d];
                }
                mixes[i * hc + h] = dot * rsqrt;
            }
        }

        // 官方实现: pre = sigmoid(mixes * hc_scale + hc_base) + eps
        // hc_head_scale 是标量 (shape=[1])，广播到所有 hc 分支
        // hc_head_base 是向量 (shape=[hc_mult])，每个分支独立偏置
        let s = scale_f32[0];
        let mut pre = vec![0.0f32; n * hc];
        for i in 0..n {
            for h in 0..hc {
                let b = base_f32[h];
                pre[i * hc + h] = 1.0 / (1.0 + (-(mixes[i * hc + h] * s + b)).exp()) + hc_eps;
            }
        }

        let mut y = vec![0.0f32; n * dim];
        for i in 0..n {
            for h in 0..hc {
                let w = pre[i * hc + h];
                for d in 0..dim {
                    y[i * dim + d] += w * x_f32[i * x_dim + h * dim + d];
                }
            }
        }

        let y_bytes: Vec<u8> = bytemuck::cast_slice(&y).to_vec();
        let y_cpu = CpuTensor::new(y_bytes, vec![bsz, seqlen, dim], DType::FP32);
        GpuTensor::from_host(device, &y_cpu)
    }

    fn rmsnorm(&self, x: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor> {
        let x_host = x.to_host()?;
        let w_host = weight.to_host()?;
        let x_f32: Vec<f32> = if x_host.dtype == DType::BF16 {
            let bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
            bf16.iter().map(|v| v.to_f32()).collect()
        } else {
            bytemuck::cast_slice(&x_host.data).to_vec()
        };
        let w_f32: Vec<f32> = if w_host.dtype == DType::BF16 {
            let bf16: &[half::bf16] = bytemuck::cast_slice(&w_host.data);
            bf16.iter().map(|v| v.to_f32()).collect()
        } else {
            bytemuck::cast_slice(&w_host.data).to_vec()
        };

        let n = x.shape[0] * x.shape[1];
        let dim = x.shape[2];
        let eps = 1e-6f32;

        let mut out = vec![0.0f32; x_f32.len()];
        for i in 0..n {
            let ss: f32 = x_f32[i * dim..(i + 1) * dim].iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv_rms = 1.0 / (ss + eps).sqrt();
            for d in 0..dim {
                out[i * dim + d] = x_f32[i * dim + d] * inv_rms * w_f32[d];
            }
        }

        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out).to_vec();
        let out_cpu = CpuTensor::new(out_bytes, x_host.shape.clone(), DType::FP32);
        GpuTensor::from_host(x.device.clone(), &out_cpu)
    }

    fn head_forward(&self, h: &GpuTensor) -> Result<GpuTensor> {
        let seqlen = h.shape[1];
        let dim = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        let h_host = h.to_host()?;
        let h_f32: &[f32] = bytemuck::cast_slice(&h_host.data);
        let offset = if seqlen > 1 { (seqlen - 1) * dim } else { 0 };
        let last_row = &h_f32[offset..offset + dim];

        let w_f32: &[f32] = bytemuck::cast_slice(&self.head_weight_cpu.data);

        let mut logits_all = vec![0.0f32; vocab];
        let chunk_size = 4096;
        for chunk_start in (0..vocab).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab);
            for r in chunk_start..chunk_end {
                let row_off = r * dim;
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += last_row[d] * w_f32[row_off + d];
                }
                logits_all[r] = dot;
            }
        }

        let logits_bytes: Vec<u8> = bytemuck::cast_slice(&logits_all).to_vec();
        let logits_cpu = CpuTensor::new(logits_bytes, vec![1, 1, vocab], DType::FP32);

        if std::env::var("DS4RS_DEBUG_LOGITS").is_ok() {
            let mut indexed: Vec<(usize, f32)> = logits_all.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("[debug] top-5 logits:");
            for (i, &(tid, score)) in indexed.iter().take(5).enumerate() {
                eprintln!("  #{}: token={} score={:.4}", i, tid, score);
            }
        } else {
            let mut indexed: Vec<(usize, f32)> = logits_all.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            static COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let c = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if c < 3 {
                eprintln!("[logits #{}] top-3: {:?}", c, &indexed[..3]);
            }
        }

        GpuTensor::from_host(self.device.clone(), &logits_cpu)
    }
}
