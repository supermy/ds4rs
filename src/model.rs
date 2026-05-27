use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::layer::TransformerLayer;
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{anyhow, Context, Result};
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
    pub head_weight_gpu: GpuTensor,
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

        // head_weight 延迟上传 GPU：层加载后再上传，避免抢占层权重的显存
        // head_weight_gpu 将在层加载完成后初始化

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

        // head_weight 上传 GPU (FP32 保证 logits 精度，cuBLAS FP32 GEMM)
        // 延迟到层加载后，避免抢占层权重的显存
        let head_weight_gpu = match GpuTensor::from_host(device.clone(), &head_weight_cpu) {
            Ok(gpu) => {
                log_tensor_gpu("head.weight (FP32)", &gpu);
                gpu
            }
            Err(e) => {
                eprintln!("[load] WARNING: head_weight GPU upload failed ({}), will use CPU fallback", e);
                GpuTensor::zeros(device.clone(), vec![0], DType::FP32)?
            }
        };
        log_vram(&device, "head.weight 上传后");

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
            head_weight_gpu,
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

    fn reshape(&self, t: &GpuTensor, new_shape: &[usize]) -> Result<GpuTensor> {
        let old_nbytes: usize = t.shape.iter().product::<usize>() * t.dtype.element_size();
        let new_nbytes: usize = new_shape.iter().product::<usize>() * t.dtype.element_size();
        if old_nbytes != new_nbytes {
            return Err(anyhow!(
                "reshape size mismatch: {:?} ({}) vs {:?} ({})",
                t.shape, old_nbytes, new_shape, new_nbytes
            ));
        }
        Ok(GpuTensor {
            slice: t.slice.clone(),
            shape: new_shape.to_vec(),
            dtype: t.dtype,
            device: t.device.clone(),
        })
    }

    fn cast_to_f32(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype == DType::FP32 {
            return Ok(x.clone());
        }
        if x.dtype != DType::BF16 {
            return Err(anyhow!("cast_to_f32: unsupported dtype {:?}", x.dtype));
        }
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            128 => Some("cast_bf16_to_f32_N128"),
            512 => Some("cast_bf16_to_f32_N512"),
            1024 => Some("cast_bf16_to_f32_N1024"),
            4096 => Some("cast_bf16_to_f32_N4096"),
            8192 => Some("cast_bf16_to_f32_N8192"),
            16384 => Some("cast_bf16_to_f32_N16384"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let m = n / last_dim;
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::FP32)?;
            self.kernels.call(kname, &[&x_2d, &y_2d])
                .with_context(|| format!("cast_to_f32 kernel {} failed", kname))?;
            return Ok(GpuTensor {
                slice: y_2d.slice,
                shape: x.shape.clone(),
                dtype: DType::FP32,
                device,
            });
        }

        // Fallback: D2H → cast → H2D (for small tensors)
        let x_host = x.to_host()?;
        let bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
        let f32_data: Vec<f32> = bf16.iter().map(|v| v.to_f32()).collect();
        let cpu = CpuTensor::new(bytemuck::cast_slice(&f32_data).to_vec(), x_host.shape.clone(), DType::FP32);
        GpuTensor::from_host(device, &cpu)
    }

    fn cast_to_bf16(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype == DType::BF16 {
            return Ok(x.clone());
        }
        if x.dtype != DType::FP32 {
            return Err(anyhow!("cast_to_bf16: unsupported dtype {:?}", x.dtype));
        }
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            128 => Some("cast_f32_to_bf16_N128"),
            512 => Some("cast_f32_to_bf16_N512"),
            1024 => Some("cast_f32_to_bf16_N1024"),
            4096 => Some("cast_f32_to_bf16_N4096"),
            16384 => Some("cast_f32_to_bf16_N16384"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let m = n / last_dim;
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::FP32,
                device: device.clone(),
            };
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::BF16)?;
            self.kernels.call(kname, &[&x_2d, &y_2d])
                .with_context(|| format!("cast_to_bf16 kernel {} failed", kname))?;
            return Ok(GpuTensor {
                slice: y_2d.slice,
                shape: x.shape.clone(),
                dtype: DType::BF16,
                device,
            });
        }

        // Fallback: D2H → cast → H2D
        let x_host = x.to_host()?;
        let f32_data: &[f32] = bytemuck::cast_slice(&x_host.data);
        let bf16_data: Vec<half::bf16> = f32_data.iter().map(|v| half::bf16::from_f32(*v)).collect();
        let cpu = CpuTensor::new(bytemuck::cast_slice(&bf16_data).to_vec(), x_host.shape.clone(), DType::BF16);
        GpuTensor::from_host(device, &cpu)
    }

    fn gemm_f32(&self, x: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor> {
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        let n = weight.shape[0];
        let device = x.device.clone();
        let mut c = GpuTensor::zeros(device, vec![m, n], DType::FP32)?;
        self.cublas.gemm_f32(m, n, k, x, weight, &mut c, 1.0, 0.0)?;
        Ok(c)
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
        let total = bsz * seqlen;
        let hc_dim = hc * dim;

        // Step 1: x BF16 → FP32, reshape [total, hc*dim]
        let x_f32 = self.cast_to_f32(x)?;
        let x_flat = self.reshape(&x_f32, &[total, hc_dim])?;

        // Step 2: rmsnorm_f32 (no weight) → x_normed
        let x_normed = {
            let kernel_name = match hc_dim {
                16384 => "rmsnorm_f32_N16384",
                4096 => "rmsnorm_f32_N4096",
                _ => {
                    return Err(anyhow!(
                        "hc_head_reduce: no rmsnorm_f32 kernel for hc*dim={}", hc_dim
                    ));
                }
            };
            let y = GpuTensor::zeros(device.clone(), vec![total, hc_dim], DType::FP32)?;
            self.kernels.call(kernel_name, &[&x_flat, &y])
                .with_context(|| format!("hc_head_reduce rmsnorm_f32 kernel {} failed", kernel_name))?;
            y
        };

        // Step 3: GEMM x_normed @ hc_head_fn^T → mixes [total, hc]
        let hc_fn_f32 = self.cast_to_f32(&self.hc_head_fn)?;
        let mixes = self.gemm_f32(&x_normed, &hc_fn_f32)?;

        // Step 4: hc_sigmoid_hc4 kernel: sigmoid(mixes * scale + base) + eps → pre [total, hc] BF16
        let hc_scale_f32 = self.cast_to_f32(&self.hc_head_scale)?;
        let hc_base_f32 = self.cast_to_f32(&self.hc_head_base)?;
        let pre_bf16 = GpuTensor::zeros(device.clone(), vec![total, hc], DType::BF16)?;
        self.kernels.call(
            "hc_sigmoid_hc4",
            &[&mixes, &hc_scale_f32, &hc_base_f32, &pre_bf16],
        ).with_context(|| "hc_sigmoid_hc4 kernel failed")?;

        // Step 5: Cast pre to FP32
        let pre_f32 = self.cast_to_f32(&pre_bf16)?;

        // Step 6: hc_reduce: y[i,d] = sum_h pre[i,h] * x[i,h,d]
        // Using strided batched GEMM: for each batch i, C[i] = pre[i] @ x[i]
        // pre_f32: [total, hc], x_f32: [total, hc, dim] → y_f32: [total, dim]
        let pre_2d = GpuTensor {
            slice: pre_f32.slice.clone(),
            shape: vec![total, hc],
            dtype: DType::FP32,
            device: device.clone(),
        };
        let x_3d = GpuTensor {
            slice: x_f32.slice.clone(),
            shape: vec![total, hc, dim],
            dtype: DType::FP32,
            device: device.clone(),
        };
        let mut y_f32 = GpuTensor::zeros(device.clone(), vec![total, dim], DType::FP32)?;
        self.cublas.gemm_f32_nn_strided_batched(
            1, dim, hc,
            &pre_2d, &x_3d, &mut y_f32,
            hc as i64, (hc * dim) as i64, dim as i64,
            total as i32,
            1.0, 0.0,
        )?;

        // Debug: 对比 GPU 和 CPU 结果（仅第一次）
        static DEBUG_COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
        let dc = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if dc < 1 {
            // CPU 参考实现
            let x_host = x.to_host()?;
            let x_cpu: Vec<f32> = if x_host.dtype == DType::BF16 {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            } else {
                bytemuck::cast_slice(&x_host.data).to_vec()
            };
            let fn_host = self.hc_head_fn.to_host()?;
            let fn_cpu: Vec<f32> = if fn_host.dtype == DType::BF16 {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&fn_host.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            } else {
                bytemuck::cast_slice(&fn_host.data).to_vec()
            };
            let scale_host = self.hc_head_scale.to_host()?;
            let scale_f32: f32 = if scale_host.dtype == DType::BF16 {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&scale_host.data);
                bf16[0].to_f32()
            } else {
                bytemuck::cast_slice(&scale_host.data)[0]
            };
            let base_host = self.hc_head_base.to_host()?;
            let base_cpu: Vec<f32> = if base_host.dtype == DType::BF16 {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&base_host.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            } else {
                bytemuck::cast_slice(&base_host.data).to_vec()
            };

            let norm_eps = 1e-6;
            let hc_eps = 1e-6;
            let mut y_cpu = vec![0.0f32; total * dim];
            for i in 0..total {
                let x_row = &x_cpu[i * hc_dim..(i + 1) * hc_dim];
                let mean_sq: f32 = x_row.iter().map(|v| v * v).sum::<f32>() / hc_dim as f32;
                let rsqrt = 1.0 / (mean_sq + norm_eps).sqrt();
                for h in 0..hc {
                    let mut dot = 0.0f32;
                    for d in 0..hc_dim {
                        dot += x_row[d] * fn_cpu[h * hc_dim + d];
                    }
                    let mix_val = dot * rsqrt;
                    let pre_val = 1.0 / (1.0 + (-mix_val * scale_f32 - base_cpu[h]).exp()) + hc_eps;
                    for d in 0..dim {
                        y_cpu[i * dim + d] += pre_val * x_cpu[i * hc_dim + h * dim + d];
                    }
                }
            }

            // GPU 结果
            let y_host = y_f32.to_host()?;
            let y_gpu: &[f32] = bytemuck::cast_slice(&y_host.data);

            // 对比
            let mut max_diff = 0.0f32;
            let mut max_diff_idx = 0;
            for (idx, (a, b)) in y_cpu.iter().zip(y_gpu.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = idx;
                }
            }
            let cpu_min = y_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
            let cpu_max = y_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let gpu_min = y_gpu.iter().cloned().fold(f32::INFINITY, f32::min);
            let gpu_max = y_gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[hc_head_reduce DEBUG] CPU: min={:.4} max={:.4} | GPU: min={:.4} max={:.4} | max_diff={:.6} at idx={}",
                cpu_min, cpu_max, gpu_min, gpu_max, max_diff, max_diff_idx);
            if max_diff > 0.01 {
                let i = max_diff_idx / dim;
                let d = max_diff_idx % dim;
                eprintln!("[hc_head_reduce DEBUG]   CPU[{},{}] = {:.6}, GPU[{},{}] = {:.6}",
                    i, d, y_cpu[max_diff_idx], i, d, y_gpu[max_diff_idx]);
                // 打印 pre 值对比
                let pre_host = pre_f32.to_host()?;
                let pre_gpu: &[f32] = bytemuck::cast_slice(&pre_host.data);
                eprintln!("[hc_head_reduce DEBUG]   pre_gpu[0] = {:?}", &pre_gpu[0..hc.min(4)]);
                // 打印 mixes
                let mixes_host = mixes.to_host()?;
                let mixes_gpu: &[f32] = bytemuck::cast_slice(&mixes_host.data);
                eprintln!("[hc_head_reduce DEBUG]   mixes_gpu[0] = {:?}", &mixes_gpu[0..hc.min(4)]);
                eprintln!("[hc_head_reduce DEBUG]   scale={:.6} base={:?}", scale_f32, &base_cpu[..hc.min(4)]);
            }
        }

        // Step 7: Cast y to BF16 → [bsz, seqlen, dim]
        let y_bf16 = self.cast_to_bf16(&y_f32)?;
        Ok(GpuTensor {
            slice: y_bf16.slice,
            shape: vec![bsz, seqlen, dim],
            dtype: DType::BF16,
            device: y_bf16.device,
        })
    }

    fn rmsnorm(&self, x: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor> {
        let device = x.device.clone();
        let shape = x.shape.clone();
        let last_dim = shape.last().copied().unwrap_or(1);

        // TileLang rmsnorm kernel: BF16 input, FP32 weight, BF16 output
        let x_bf16 = if x.dtype == DType::BF16 {
            x.clone()
        } else {
            self.cast_to_bf16(x)?
        };
        let w_f32 = if weight.dtype == DType::FP32 {
            weight.clone()
        } else {
            self.cast_to_f32(weight)?
        };

        let kernel_name = match last_dim {
            4096 => "rmsnorm_N4096",
            1024 => "rmsnorm_N1024",
            512 => "rmsnorm_N512",
            _ => {
                return Err(anyhow!(
                    "rmsnorm: no GPU kernel for last_dim={}", last_dim
                ));
            }
        };

        let n_rows: usize = shape.iter().rev().skip(1).product();
        let x_2d = GpuTensor {
            slice: x_bf16.slice.clone(),
            shape: vec![n_rows, last_dim],
            dtype: DType::BF16,
            device: device.clone(),
        };
        let y = GpuTensor::zeros(device.clone(), vec![n_rows, last_dim], DType::BF16)?;
        self.kernels.call(kernel_name, &[&x_2d, &w_f32, &y])
            .with_context(|| format!("rmsnorm kernel {} failed", kernel_name))?;

        Ok(GpuTensor {
            slice: y.slice,
            shape,
            dtype: DType::BF16,
            device,
        })
    }

    fn head_forward(&self, h: &GpuTensor) -> Result<GpuTensor> {
        let seqlen = h.shape[1];
        let dim = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let device = h.device.clone();
        let bsz = h.shape[0];

        // 检查 head_weight_gpu 是否可用（shape=[0] 表示上传失败，回退 CPU）
        let gpu_available = self.head_weight_gpu.shape[0] == vocab;

        if gpu_available {
            // GPU 路径: cuBLAS FP32 GEMM (head_weight_gpu 是 FP32)
            let h_f32 = self.cast_to_f32(h)?;

            let h_last = if seqlen > 1 {
                let h_host = h_f32.to_host()?;
                let row_bytes = dim * 4; // FP32
                let offset = (seqlen - 1) * row_bytes;
                let mut last_row = vec![0u8; bsz * row_bytes];
                for b in 0..bsz {
                    let src_off = b * seqlen * row_bytes + offset;
                    let dst_off = b * row_bytes;
                    last_row[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&h_host.data[src_off..src_off + row_bytes]);
                }
                let cpu = CpuTensor::new(last_row, vec![bsz, dim], DType::FP32);
                GpuTensor::from_host(device.clone(), &cpu)?
            } else {
                GpuTensor { slice: h_f32.slice.clone(), shape: vec![bsz, dim], dtype: DType::FP32, device: device.clone() }
            };

            let mut logits = GpuTensor::zeros(device, vec![bsz, vocab], DType::FP32)?;
            self.cublas.gemm_f32(bsz, vocab, dim, &h_last, &self.head_weight_gpu, &mut logits, 1.0, 0.0)?;

            // Debug logits
            {
                let logits_host = logits.to_host()?;
                let logits_f32: &[f32] = bytemuck::cast_slice(&logits_host.data);
                let mut indexed: Vec<(usize, f32)> = logits_f32.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                static COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
                let c = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < 3 || std::env::var("DS4RS_DEBUG_LOGITS").is_ok() {
                    let n = if std::env::var("DS4RS_DEBUG_LOGITS").is_ok() { 5 } else { 3 };
                    eprintln!("[logits GPU #{}] top-{}: {:?}", c, n, &indexed[..n.min(indexed.len())]);
                }
            }

            Ok(GpuTensor { slice: logits.slice, shape: vec![1, 1, vocab], dtype: DType::FP32, device: logits.device })
        } else {
            // CPU 回退: head_weight 上传 GPU 失败时使用
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

            {
                let mut indexed: Vec<(usize, f32)> = logits_all.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                static COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
                let c = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < 3 || std::env::var("DS4RS_DEBUG_LOGITS").is_ok() {
                    let n = if std::env::var("DS4RS_DEBUG_LOGITS").is_ok() { 5 } else { 3 };
                    eprintln!("[logits CPU #{}] top-{}: {:?}", c, n, &indexed[..n.min(indexed.len())]);
                }
            }

            let logits_bytes: Vec<u8> = bytemuck::cast_slice(&logits_all).to_vec();
            let logits_cpu = CpuTensor::new(logits_bytes, vec![1, 1, vocab], DType::FP32);
            GpuTensor::from_host(self.device.clone(), &logits_cpu)
        }
    }
}
