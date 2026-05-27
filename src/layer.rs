use crate::compressor::Compressor;
use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::expert::ExpertScheduler;
use crate::gate::Gate;
use crate::indexer::Indexer;
use crate::kv_cache::KvCache;
use crate::quant;
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaContext, DevicePtr};
use std::sync::Arc;

fn fmt_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} B", bytes)
    }
}

/// 简单 f32 矩阵向量乘法（CPU 标量回退）
fn matvec_f32(weight: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0f32;
        for col in 0..in_dim {
            sum += weight[row * in_dim + col] * x[col];
        }
        out[row] = sum;
    }
    out
}

fn log_vram(device: &Arc<CudaContext>, label: &str) {
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let result = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let used = total - free;
            eprintln!("[vram] {} 已用 {} / {} (剩余 {})", label, fmt_bytes(used), fmt_bytes(total), fmt_bytes(free));
        }
    }
}

fn trim_cuda_pool() {
    unsafe {
        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
        let result = cudarc::driver::sys::cuDeviceGetMemPool(&mut pool, 0);
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let _ = cudarc::driver::sys::cuMemPoolTrimTo(pool, 0);
        }
    }
}

pub struct LayerWeights {
    pub wq_a: GpuTensor,
    pub wq_a_s: GpuTensor,
    pub wq_b: GpuTensor,
    pub wq_b_s: GpuTensor,
    pub wkv: GpuTensor,
    pub wkv_s: GpuTensor,
    pub wo_a: Option<GpuTensor>,
    pub wo_a_s: GpuTensor,
    pub wo_a_dequant: CpuTensor,
    pub wo_a_gpu: GpuTensor,
    pub wo_b: GpuTensor,
    pub wo_b_s: GpuTensor,
    pub q_norm: GpuTensor,
    pub kv_norm: GpuTensor,
    pub attn_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub attn_sink: GpuTensor,
    pub hc_attn_fn: GpuTensor,
    pub hc_attn_scale: GpuTensor,
    pub hc_attn_base: GpuTensor,
    pub hc_ffn_fn: GpuTensor,
    pub hc_ffn_scale: GpuTensor,
    pub hc_ffn_base: GpuTensor,
    pub shared_w1: GpuTensor,
    pub shared_w1_s: GpuTensor,
    pub shared_w3: GpuTensor,
    pub shared_w3_s: GpuTensor,
    pub shared_w2: GpuTensor,
    pub shared_w2_s: GpuTensor,
    pub gate_weight: GpuTensor,
    pub gate_bias: Option<GpuTensor>,
    pub wo_a_is_bf16: bool,
}

pub struct TransformerLayer {
    pub layer_id: usize,
    pub weights: LayerWeights,
    pub kv_cache: KvCache,
    pub gate: Gate,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
    pub expert_scheduler: ExpertScheduler,
    pub config: Arc<ModelConfig>,
    pub kernels: Arc<KernelRegistry>,
    pub cublas: Arc<CublasHandle>,
    weight_loader: WeightLoader,
    decode_topk_cache: Option<GpuTensor>,
}

impl TransformerLayer {
    pub fn new(
        layer_id: usize,
        device: Arc<CudaContext>,
        config: Arc<ModelConfig>,
        kernels: Arc<KernelRegistry>,
        cublas: Arc<CublasHandle>,
        max_batch: usize,
        max_seqlen: usize,
        loader: &mut WeightLoader,
    ) -> Result<Self> {
        let w = Self::load_weights(layer_id, device.clone(), config.as_ref(), loader)?;
        let kv_cache = KvCache::new(device.clone(), &config, layer_id, max_batch, max_seqlen)?;

        let gate = Gate::new(
            &config,
            layer_id,
            device.clone(),
            Arc::clone(&cublas),
            Arc::clone(&kernels),
            w.gate_weight.clone(),
            w.gate_bias.clone(),
            None,
        );

        let compress_ratio = config.compress_ratio(layer_id) as usize;
        let compressor = if compress_ratio > 0 {
            match Compressor::new(
                device.clone(), &config, layer_id, compress_ratio,
                config.head_dim, false, max_batch, loader,
                cublas.clone(), Arc::clone(&kernels),
            ) {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!("[load]     warning: compressor load failed for layer {}: {}, skipping", layer_id, e);
                    None
                }
            }
        } else {
            None
        };

        let indexer = if compress_ratio > 0 && compress_ratio <= 4 {
            match Indexer::new(
                device.clone(), &config, layer_id, compress_ratio,
                max_batch, config.max_position_embeddings, loader,
                cublas.clone(), Arc::clone(&kernels),
            ) {
                Ok(i) => Some(i),
                Err(e) => {
                    eprintln!("[load]     warning: indexer load failed for layer {}: {}, skipping", layer_id, e);
                    None
                }
            }
        } else {
            None
        };

        let weight_loader = WeightLoader::from_dir(&config.model_dir)?;

        Ok(Self {
            layer_id,
            weights: w,
            kv_cache,
            gate,
            compressor,
            indexer,
            expert_scheduler: ExpertScheduler::new(device.clone(), Arc::clone(&config)),
            config,
            kernels,
            cublas,
            weight_loader,
            decode_topk_cache: None,
        })
    }

    fn load_weights(
        layer_id: usize,
        device: Arc<CudaContext>,
        config: &ModelConfig,
        loader: &mut WeightLoader,
    ) -> Result<LayerWeights> {
        let p = format!("layers.{}.", layer_id);

        let load_gpu = |loader: &mut WeightLoader, name: &str| -> Result<GpuTensor> {
            let cpu = loader
                .load(name)
                .with_context(|| format!("failed to load weight {}", name))?;
            let size_mb = cpu.data.len() as f64 / (1024.0 * 1024.0);
            eprintln!("[load]     {} {:?} {:?} ({:.2} MB)", name, cpu.shape, cpu.dtype, size_mb);
            GpuTensor::from_host(device.clone(), &cpu)
                .with_context(|| format!("failed to upload weight {}", name))
        };

        let load_optional = |loader: &mut WeightLoader, name: &str| -> Result<Option<GpuTensor>> {
            if loader.contains(name) {
                Ok(Some(load_gpu(loader, name)?))
            } else {
                Ok(None)
            }
        };

        eprintln!("[load]   ── 注意力权重 ──");
        let wq_a = load_gpu(loader, &(p.clone() + "attn.wq_a.weight"))?;
        let wq_a_s = load_optional(loader, &(p.clone() + "attn.wq_a.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });

        let wq_b = load_gpu(loader, &(p.clone() + "attn.wq_b.weight"))?;
        let wq_b_s = load_optional(loader, &(p.clone() + "attn.wq_b.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });

        let wkv = load_gpu(loader, &(p.clone() + "attn.wkv.weight"))?;
        let wkv_s = load_optional(loader, &(p.clone() + "attn.wkv.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });

        let wo_a_raw = load_gpu(loader, &(p.clone() + "attn.wo_a.weight"))?;
        let wo_a_s = load_optional(loader, &(p.clone() + "attn.wo_a.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });
        let wo_a_is_bf16 = wo_a_raw.dtype == DType::BF16
            && wo_a_raw.shape.len() == 3
            && wo_a_raw.shape[0] == config.o_groups;

        let wo_a_dequant = if wo_a_is_bf16 {
            wo_a_raw.to_host()?
        } else {
            let wo_a_host = wo_a_raw.to_host()?;
            let s_host = wo_a_s.to_host()?;
            crate::quant::dequant_fp8_e4m3_to_bf16(&wo_a_host.data, &s_host.data, &wo_a_host.shape)?
        };

        let wo_a_gpu = GpuTensor::from_host(device.clone(), &wo_a_dequant)?;
        drop(wo_a_raw);
        let wo_a: Option<GpuTensor> = None;

        let wo_b = load_gpu(loader, &(p.clone() + "attn.wo_b.weight"))?;
        let wo_b_s = load_optional(loader, &(p.clone() + "attn.wo_b.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });

        let q_norm = load_gpu(loader, &(p.clone() + "attn.q_norm.weight"))?;
        let kv_norm = load_gpu(loader, &(p.clone() + "attn.kv_norm.weight"))?;
        let attn_norm = load_gpu(loader, &(p.clone() + "attn_norm.weight"))?;
        let ffn_norm = load_gpu(loader, &(p.clone() + "ffn_norm.weight"))?;
        let attn_sink = load_gpu(loader, &(p.clone() + "attn.attn_sink"))?;

        eprintln!("[load]   ── HC 权重 ──");
        let hc_attn_fn = load_gpu(loader, &(p.clone() + "hc_attn_fn"))?;
        let hc_attn_scale = load_gpu(loader, &(p.clone() + "hc_attn_scale"))?;
        let hc_attn_base = load_gpu(loader, &(p.clone() + "hc_attn_base"))?;
        let hc_ffn_fn = load_gpu(loader, &(p.clone() + "hc_ffn_fn"))?;
        let hc_ffn_scale = load_gpu(loader, &(p.clone() + "hc_ffn_scale"))?;
        let hc_ffn_base = load_gpu(loader, &(p.clone() + "hc_ffn_base"))?;

        eprintln!("[load]   ── 共享专家 + Gate ──");
        let shared_w1 = load_gpu(loader, &(p.clone() + "ffn.shared_experts.w1.weight"))?;
        let shared_w1_s = load_optional(loader, &(p.clone() + "ffn.shared_experts.w1.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });
        let shared_w3 = load_gpu(loader, &(p.clone() + "ffn.shared_experts.w3.weight"))?;
        let shared_w3_s = load_optional(loader, &(p.clone() + "ffn.shared_experts.w3.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });
        let shared_w2 = load_gpu(loader, &(p.clone() + "ffn.shared_experts.w2.weight"))?;
        let shared_w2_s = load_optional(loader, &(p.clone() + "ffn.shared_experts.w2.scale"))?
            .unwrap_or_else(|| {
                GpuTensor::zeros(device.clone(), vec![1], DType::FP8E8M0).expect("default scale alloc")
            });

        let gate_weight = load_gpu(loader, &(p.clone() + "ffn.gate.weight"))?;
        let gate_bias = load_optional(loader, &(p.clone() + "ffn.gate.bias"))?;

        Ok(LayerWeights {
            wq_a, wq_a_s, wq_b, wq_b_s, wkv, wkv_s,
            wo_a, wo_a_s, wo_a_dequant, wo_a_gpu, wo_b, wo_b_s,
            q_norm, kv_norm, attn_norm, ffn_norm, attn_sink,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            shared_w1, shared_w1_s, shared_w3, shared_w3_s, shared_w2, shared_w2_s,
            gate_weight, gate_bias, wo_a_is_bf16,
        })
    }

    fn debug_tensor_stats(label: &str, x: &GpuTensor, layer_id: usize) {
        let Ok(h) = x.to_host() else { return; };
        let f32_data: Vec<f32> = if h.dtype == DType::BF16 {
            let bf16: &[half::bf16] = bytemuck::cast_slice(&h.data);
            bf16.iter().map(|v| v.to_f32()).collect()
        } else if h.dtype == DType::FP32 {
            bytemuck::cast_slice(&h.data).to_vec()
        } else if h.dtype == DType::FP8E8M0 {
            let raw: &[u8] = bytemuck::cast_slice(&h.data);
            raw.iter().map(|&b| quant::e8m0_to_f32(b)).collect()
        } else if h.dtype == DType::FP8E4M3 {
            let raw: &[u8] = bytemuck::cast_slice(&h.data);
            raw.iter().map(|&b| quant::fp8_e4m3_to_f32(b)).collect()
        } else {
            return;
        };
        let mean = f32_data.iter().sum::<f32>() / f32_data.len() as f32;
        let max = f32_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = f32_data.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[L{:02}] {:40} shape={:?} mean={:.6e} min={:.6e} max={:.6e}", layer_id, label, x.shape, mean, min, max);
    }

    pub fn forward(
        &mut self,
        x: &GpuTensor,
        start_pos: usize,
        rope: &RopeCache,
        input_ids: Option<&[u32]>,
    ) -> Result<GpuTensor> {
        self.expert_scheduler.finalize_prefetch();

        let lid = self.layer_id;
        let debug = lid == 0 || lid == 10 || lid == 20 || lid == 30 || lid == 40 || lid == 42;

        let residual = x.clone();

        let (x_attn, post, comb) =
            self.hc_pre(x, &self.weights.hc_attn_fn, &self.weights.hc_attn_scale, &self.weights.hc_attn_base)?;

        let x_norm = self.rmsnorm(&x_attn, Some(&self.weights.attn_norm))?;
        let attn_out = self.attention(&x_norm, start_pos, rope)?;

        let x = self.hc_post(&attn_out, &residual, &post, &comb)?;

        if debug {
            Self::debug_tensor_stats("after hc_post(attn)", &x, lid);
            Self::debug_tensor_stats("post(attn)", &post, lid);
        }

        drop(residual);
        drop(post);
        drop(comb);
        drop(x_attn);
        drop(x_norm);
        drop(attn_out);

        let residual = x.clone();

        let (x_ffn, post, comb) =
            self.hc_pre(&x, &self.weights.hc_ffn_fn, &self.weights.hc_ffn_scale, &self.weights.hc_ffn_base)?;

        let x_norm = self.rmsnorm(&x_ffn, Some(&self.weights.ffn_norm))?;
        let ffn_out = self.ffn(&x_norm, input_ids)?;

        let x = self.hc_post(&ffn_out, &residual, &post, &comb)?;

        if debug {
            Self::debug_tensor_stats("after hc_post(ffn)", &x, lid);
            Self::debug_tensor_stats("post(ffn)", &post, lid);
        }

        drop(residual);
        drop(post);
        drop(comb);
        drop(x_ffn);
        drop(x_norm);
        drop(ffn_out);

        self.expert_scheduler.clear_gpu_cache();
        trim_cuda_pool();

        Ok(x)
    }

    fn hc_reduce(&self, x: &GpuTensor, pre: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let hc = x.shape[2];
        let dim = x.shape[3];
        let device = x.device.clone();
        let total = bsz * seqlen;

        let pre_2d = GpuTensor {
            slice: pre.slice.clone(),
            shape: vec![total, hc],
            dtype: DType::FP32,
            device: device.clone(),
        };

        let x_f32 = self.cast_to_f32(&GpuTensor {
            slice: x.slice.clone(),
            shape: vec![total, hc, dim],
            dtype: x.dtype,
            device: device.clone(),
        })?;

        let mut y_f32 = GpuTensor::zeros(device.clone(), vec![total, dim], DType::FP32)?;
        self.cublas.gemm_f32_nn_strided_batched(
            1, dim, hc,
            &pre_2d, &x_f32, &mut y_f32,
            hc as i64, (hc * dim) as i64, dim as i64,
            total as i32,
            1.0, 0.0,
        )?;

        if self.layer_id == 0 || self.layer_id == 42 {
            Self::debug_tensor_stats("hc_reduce_y", &y_f32, self.layer_id);
        }

        let y = self.cast_to_bf16(&y_f32)?;
        Ok(GpuTensor { slice: y.slice, shape: vec![bsz, seqlen, dim], dtype: DType::BF16, device })
    }

    fn hc_pre(
        &self,
        x: &GpuTensor,
        hc_fn: &GpuTensor,
        hc_scale: &GpuTensor,
        hc_base: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let hc = self.config.hc_mult;
        let device = x.device.clone();
        let total = bsz * seqlen;
        let mix_hc = (2 + hc) * hc;

        if x.shape.len() != 4 || x.shape[2] != hc {
            return Err(anyhow!("hc_pre: x shape must be [*, *, {}, *], got {:?}", hc, x.shape));
        }
        if hc_fn.shape.len() != 2 || hc_fn.shape[0] != mix_hc || hc_fn.shape[1] != hc * self.config.hidden_size {
            return Err(anyhow!(
                "hc_pre: hc_fn shape must be [{}, {}], got {:?}",
                mix_hc, hc * self.config.hidden_size, hc_fn.shape
            ));
        }
        if hc_scale.shape.len() != 1 || hc_scale.shape[0] != 3 {
            return Err(anyhow!("hc_pre: hc_scale shape must be [3], got {:?}", hc_scale.shape));
        }
        if hc_base.shape.len() != 1 || hc_base.shape[0] != mix_hc {
            return Err(anyhow!("hc_pre: hc_base shape must be [{}], got {:?}", mix_hc, hc_base.shape));
        }

        let x_flat = self.reshape(x, &[total, hc * self.config.hidden_size])?;
        let x_flat_f32 = self.cast_to_f32(&x_flat)?;

        let x_normed = self.rmsnorm_f32(&x_flat_f32)?;

        let hc_fn_f32 = self.cast_to_f32(hc_fn)?;
        let mixes = self.gemm_f32(&x_normed, &hc_fn_f32)?;

        if self.layer_id == 0 || self.layer_id == 42 {
            Self::debug_tensor_stats("hc_pre_x_normed", &x_normed, self.layer_id);
            Self::debug_tensor_stats("hc_pre_mixes", &mixes, self.layer_id);
        }

        let hc_scale_f32 = self.cast_to_f32(hc_scale)?;
        let hc_base_f32 = self.cast_to_f32(hc_base)?;

        let mixes_2d = GpuTensor {
            slice: mixes.slice.clone(),
            shape: vec![total, mix_hc],
            dtype: mixes.dtype,
            device: device.clone(),
        };

        let pre_out = GpuTensor::zeros(device.clone(), vec![total, hc], DType::FP32)?;
        let post_out = GpuTensor::zeros(device.clone(), vec![total, hc], DType::FP32)?;
        let comb_out = GpuTensor::zeros(device.clone(), vec![total, hc, hc], DType::FP32)?;

        self.kernels.call(
            "hc_sinkhorn_hc4_it20",
            &[&mixes_2d, &hc_scale_f32, &hc_base_f32, &pre_out, &post_out, &comb_out],
        ).with_context(|| "hc_sinkhorn_hc4_it20 kernel call failed")?;

        let pre = GpuTensor { slice: pre_out.slice, shape: vec![bsz, seqlen, hc], dtype: DType::FP32, device: device.clone() };
        let post = GpuTensor { slice: post_out.slice, shape: vec![bsz, seqlen, hc], dtype: DType::FP32, device: device.clone() };
        let comb = GpuTensor { slice: comb_out.slice, shape: vec![bsz, seqlen, hc, hc], dtype: DType::FP32, device: device.clone() };

        if self.layer_id == 0 || self.layer_id == 42 {
            Self::debug_tensor_stats("hc_pre_pre", &pre, self.layer_id);
            Self::debug_tensor_stats("hc_pre_post", &post, self.layer_id);
            Self::debug_tensor_stats("hc_pre_comb", &comb, self.layer_id);
        }

        let y = self.hc_reduce(x, &pre)?;
        Ok((y, post, comb))
    }

    fn hc_post(
        &self,
        x: &GpuTensor,
        residual: &GpuTensor,
        post: &GpuTensor,
        comb: &GpuTensor,
    ) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let hc = self.config.hc_mult;
        let dim = self.config.hidden_size;
        let device = x.device.clone();
        let total = bsz * seqlen;

        let post_2d = GpuTensor {
            slice: post.slice.clone(),
            shape: vec![total, hc],
            dtype: DType::FP32,
            device: device.clone(),
        };

        let comb_3d = GpuTensor {
            slice: comb.slice.clone(),
            shape: vec![total, hc, hc],
            dtype: DType::FP32,
            device: device.clone(),
        };

        let x_f32 = self.cast_to_f32(&GpuTensor {
            slice: x.slice.clone(),
            shape: vec![total, dim],
            dtype: x.dtype,
            device: device.clone(),
        })?;

        let residual_f32 = self.cast_to_f32(&GpuTensor {
            slice: residual.slice.clone(),
            shape: vec![total, hc, dim],
            dtype: residual.dtype,
            device: device.clone(),
        })?;

        let mut y_f32 = GpuTensor::zeros(device.clone(), vec![total, hc, dim], DType::FP32)?;

        self.cublas.gemm_f32_nn_strided_batched(
            hc, dim, 1,
            &post_2d, &x_f32, &mut y_f32,
            hc as i64, dim as i64, (hc * dim) as i64,
            total as i32,
            1.0, 0.0,
        )?;

        self.cublas.gemm_f32_tn_strided_batched(
            hc, dim, hc,
            &comb_3d, &residual_f32, &mut y_f32,
            (hc * hc) as i64, (hc * dim) as i64, (hc * dim) as i64,
            total as i32,
            1.0, 1.0,
        )?;

        if self.layer_id == 0 || self.layer_id == 42 {
            Self::debug_tensor_stats("hc_post_comb", &comb_3d, self.layer_id);
            Self::debug_tensor_stats("hc_post_residual", &residual_f32, self.layer_id);
            Self::debug_tensor_stats("hc_post_x", &x_f32, self.layer_id);
            Self::debug_tensor_stats("hc_post_post", &post_2d, self.layer_id);
            Self::debug_tensor_stats("hc_post_final_y", &y_f32, self.layer_id);

            let y_host = y_f32.to_host().unwrap();
            let y_data: &[f32] = bytemuck::cast_slice(&y_host.data);
            let comb_host = comb_3d.to_host().unwrap();
            let comb_data: &[f32] = bytemuck::cast_slice(&comb_host.data);
            let res_host = residual_f32.to_host().unwrap();
            let res_data: &[f32] = bytemuck::cast_slice(&res_host.data);
            let x_host = x_f32.to_host().unwrap();
            let x_data: &[f32] = bytemuck::cast_slice(&x_host.data);
            let post_host = post_2d.to_host().unwrap();
            let post_data: &[f32] = bytemuck::cast_slice(&post_host.data);
            let mut max_diff = 0.0f32;
            let mut max_diff_dst = 0;
            let mut max_diff_d = 0;
            for t in 0..total.min(1) {
                for dst in 0..hc {
                    for d in 0..dim.min(16) {
                        let mut acc = x_data[t * dim + d] * post_data[t * hc + dst];
                        for src in 0..hc {
                            acc += comb_data[t * hc * hc + src * hc + dst] * res_data[t * hc * dim + src * dim + d];
                        }
                        let gpu_val = y_data[t * hc * dim + dst * dim + d];
                        let diff = (gpu_val - acc).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            max_diff_dst = dst;
                            max_diff_d = d;
                        }
                    }
                }
            }
            eprintln!("[L{:02}] hc_post CPU vs GPU max_diff={:.6} at dst={}, d={}", self.layer_id, max_diff, max_diff_dst, max_diff_d);

            // 验证 comb 矩阵的双随机性
            for t in 0..total.min(1) {
                for row in 0..hc {
                    let row_sum: f32 = (0..hc).map(|col| comb_data[t * hc * hc + row * hc + col]).sum();
                    let col_sum: f32 = (0..hc).map(|col| comb_data[t * hc * hc + col * hc + row]).sum();
                    if row == 0 {
                        eprintln!("[L{:02}] comb row_sum[0]={:.6} col_sum[0]={:.6}", self.layer_id, row_sum, col_sum);
                    }
                }
            }
        }

        let y = self.cast_to_bf16(&y_f32)?;
        Ok(GpuTensor { slice: y.slice, shape: vec![bsz, seqlen, hc, dim], dtype: DType::BF16, device })
    }

    fn attention(
        &mut self,
        x: &GpuTensor,
        start_pos: usize,
        rope: &RopeCache,
    ) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let n_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = self.config.kv_dim();
        let win = self.config.sliding_window;
        let ratio = self.config.compress_ratio(self.layer_id) as usize;

        let lid = self.layer_id;
        let debug_attn = {
            static C: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let c = C.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            c < 2 && (lid == 0 || lid == 42)
        };

        let qr = self.fp8_gemm_act_quant(x, &self.weights.wq_a, &self.weights.wq_a_s, self.config.q_lora_rank)?;
        let qr = self.reshape(&qr, &[bsz, seqlen, self.config.q_lora_rank])?;
        let q_normed = self.rmsnorm(&qr, Some(&self.weights.q_norm))?;
        let q_normed_flat = self.reshape(&q_normed, &[bsz * seqlen, self.config.q_lora_rank])?;
        let q = self.fp8_gemm_act_quant(&q_normed_flat, &self.weights.wq_b, &self.weights.wq_b_s, n_heads * head_dim)?;

        let q = self.reshape(&q, &[bsz, seqlen, n_heads, head_dim])?;
        let q = self.rmsnorm(&q, None)?;

        if debug_attn { Self::debug_tensor_stats("q_after_norm", &q, lid); }

        let q_rotated = self.apply_rope_q(&q, start_pos, rope)?;

        let kv = self.fp8_gemm_act_quant(x, &self.weights.wkv, &self.weights.wkv_s, kv_dim)?;
        let kv = self.reshape(&kv, &[bsz, seqlen, kv_dim])?;
        let kv_normed = self.rmsnorm(&kv, Some(&self.weights.kv_norm))?;
        let kv_rotated = self.apply_rope_kv(&kv_normed, start_pos, rope)?;
        let kv_final = self.act_quant_inplace_nope(&kv_rotated)?;

        if debug_attn { Self::debug_tensor_stats("kv_final", &kv_final, lid); }

        let swa_topk = self.compute_topk_idxs(bsz, seqlen, start_pos)?;

        let (topk_idxs, attn_kv) = if ratio > 0 {
            let offset = if start_pos == 0 { seqlen } else { win };

            let compress_topk = if let Some(ref mut indexer) = self.indexer {
                indexer.forward(x, &q_normed, start_pos, offset, bsz, seqlen)?
            } else {
                self.get_compress_topk_uniform(ratio, bsz, seqlen, start_pos, offset)?
            };

            let combined = self.concat_topk(&swa_topk, &compress_topk, bsz, seqlen)?;

            if start_pos == 0 {
                self.kv_cache.update_prefill(&kv_final, 0, seqlen)?;

                let compressed = if let Some(ref mut compressor) = self.compressor {
                    compressor.forward(x, start_pos, bsz, seqlen)?
                } else {
                    None
                };

                if let Some(ref comp_kv) = compressed {
                    let _ = self.kv_cache.write_compressed(comp_kv, 0, start_pos, seqlen);
                    let full_kv = self.concat_kv(&kv_final, comp_kv, bsz, seqlen)?;
                    (combined, full_kv)
                } else {
                    (combined, self.kv_cache.get_full_cache(0)?)
                }
            } else {
                self.kv_cache.update_decode(&kv_final, 0, start_pos)?;

                let compressed = if let Some(ref mut compressor) = self.compressor {
                    compressor.forward(x, start_pos, bsz, seqlen)?
                } else {
                    None
                };

                if let Some(ref comp_kv) = compressed {
                    let _ = self.kv_cache.write_compressed(comp_kv, 0, start_pos, 1);
                }

                (combined, self.kv_cache.get_full_cache(0)?)
            }
        } else {
            if start_pos == 0 {
                self.kv_cache.update_prefill(&kv_final, 0, seqlen)?;
            } else {
                self.kv_cache.update_decode(&kv_final, 0, start_pos)?;
            }

            (swa_topk, self.kv_cache.get_full_cache(0)?)
        };

        let attn_out = self.sparse_attention(&q_rotated, &attn_kv, &self.weights.attn_sink, &topk_idxs, bsz, seqlen, start_pos)?;
        if debug_attn { Self::debug_tensor_stats("sparse_attn_out", &attn_out, lid); }
        let attn_out = self.apply_inverse_rope(&attn_out, start_pos, rope)?;
        if debug_attn { Self::debug_tensor_stats("after_inv_rope", &attn_out, lid); }
        self.output_proj(&attn_out)
    }

    fn output_proj(&self, o: &GpuTensor) -> Result<GpuTensor> {
        let bsz = o.shape[0];
        let seqlen = o.shape[1];
        let n_groups = self.config.o_groups;
        let o_lora_rank = self.config.o_lora_rank;
        let group_dim = self.config.num_attention_heads * self.config.head_dim / n_groups;
        let dim = self.config.hidden_size;
        let device = o.device.clone();
        let total = bsz * seqlen;

        let lid = self.layer_id;
        let debug_op = {
            static C: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            let c = C.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            c < 2 && (lid == 0 || lid == 42)
        };

        if debug_op { Self::debug_tensor_stats("sparse_attn_out(before_proj)", o, lid); }

        let mut result = GpuTensor::zeros(device.clone(), vec![total, n_groups * o_lora_rank], DType::BF16)?;

        for g in 0..n_groups {
            let a_elem_offset = g * o_lora_rank * group_dim;
            let b_elem_offset = g * group_dim;
            let c_elem_offset = g * o_lora_rank;

            self.cublas.gemm_bf16_tn(
                o_lora_rank, total, group_dim,
                &self.weights.wo_a_gpu, group_dim as i32, a_elem_offset,
                o, (n_groups * group_dim) as i32, b_elem_offset,
                &mut result, (n_groups * o_lora_rank) as i32, c_elem_offset,
                1.0, 0.0,
            )?;
        }

        if debug_op { Self::debug_tensor_stats("after_wo_a", &result, lid); }
        if debug_op { Self::debug_tensor_stats("wo_b_scale", &self.weights.wo_b_s, lid); }
        if debug_op { Self::debug_tensor_stats("wo_b_weight", &self.weights.wo_b, lid); }

        let x = self.fp8_gemm_act_quant(&result, &self.weights.wo_b, &self.weights.wo_b_s, dim)?;

        if debug_op { Self::debug_tensor_stats("after_wo_b", &x, lid); }

        self.reshape(&x, &[bsz, seqlen, dim])
    }

    fn ffn_shared(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = self.config.hidden_size;
        let inter_dim = self.config.moe_intermediate_size;
        let device = x.device.clone();
        let total = bsz * seqlen;

        let x_flat = self.reshape(x, &[total, dim])?;
        let gate = self.fp8_gemm_act_quant(&x_flat, &self.weights.shared_w1, &self.weights.shared_w1_s, inter_dim)?;
        let up = self.fp8_gemm_act_quant(&x_flat, &self.weights.shared_w3, &self.weights.shared_w3_s, inter_dim)?;

        if self.layer_id == 42 {
            Self::debug_tensor_stats("ffn_gate", &gate, self.layer_id);
            Self::debug_tensor_stats("ffn_up", &up, self.layer_id);
        }

        if false && dim == 4096 && inter_dim == 2048 && self.kernels.has("fused_shared_ffn_D4096_I2048") {
            if gate.dtype != DType::BF16 || up.dtype != DType::BF16 {
                return Err(anyhow!("fused_shared_ffn: gate/up must be BF16, got gate={:?} up={:?}", gate.dtype, up.dtype));
            }
            if gate.shape.len() != 2 || gate.shape[0] != total || gate.shape[1] != inter_dim {
                return Err(anyhow!("fused_shared_ffn: gate shape must be [{}, {}], got {:?}", total, inter_dim, gate.shape));
            }
            if up.shape != gate.shape {
                return Err(anyhow!("fused_shared_ffn: up shape {:?} != gate shape {:?}", up.shape, gate.shape));
            }
            if self.weights.shared_w2.dtype != DType::FP8E4M3 {
                return Err(anyhow!("fused_shared_ffn: w2 dtype must be FP8E4M3, got {:?}", self.weights.shared_w2.dtype));
            }
            if self.weights.shared_w2.shape.len() != 2 || self.weights.shared_w2.shape[0] != dim || self.weights.shared_w2.shape[1] != inter_dim {
                return Err(anyhow!(
                    "fused_shared_ffn: w2 shape must be [{}, {}], got {:?}",
                    dim, inter_dim, self.weights.shared_w2.shape
                ));
            }
            let expected_ws_rows = (dim + 127) / 128;
            let expected_ws_cols = (inter_dim + 127) / 128;
            if self.weights.shared_w2_s.shape.len() != 2 || self.weights.shared_w2_s.shape[0] != expected_ws_rows || self.weights.shared_w2_s.shape[1] != expected_ws_cols {
                return Err(anyhow!(
                    "fused_shared_ffn: w2_scale shape must be [{}, {}], got {:?}",
                    expected_ws_rows, expected_ws_cols, self.weights.shared_w2_s.shape
                ));
            }

            let gate_2d = GpuTensor {
                slice: gate.slice.clone(),
                shape: vec![total, inter_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let up_2d = GpuTensor {
                slice: up.slice.clone(),
                shape: vec![total, inter_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let w2_2d = GpuTensor {
                slice: self.weights.shared_w2.slice.clone(),
                shape: vec![dim, inter_dim],
                dtype: DType::FP8E4M3,
                device: device.clone(),
            };
            let w2_s_2d = GpuTensor {
                slice: self.weights.shared_w2_s.slice.clone(),
                shape: self.weights.shared_w2_s.shape.clone(),
                dtype: self.weights.shared_w2_s.dtype,
                device: device.clone(),
            };
            let y = GpuTensor::zeros(device.clone(), vec![total, dim], DType::BF16)?;

            self.kernels.call(
                "fused_shared_ffn_D4096_I2048",
                &[&gate_2d, &up_2d, &w2_2d, &w2_s_2d, &y],
            ).with_context(|| "fused_shared_ffn kernel call failed")?;

            return self.reshape(&y, &[bsz, seqlen, dim]);
        }

        let swiglu_out = self.swiglu(&gate, &up)?;
        if self.layer_id == 42 {
            Self::debug_tensor_stats("ffn_swiglu", &swiglu_out, self.layer_id);
        }
        let down = self.fp8_gemm_act_quant(&swiglu_out, &self.weights.shared_w2, &self.weights.shared_w2_s, dim)?;
        self.reshape(&down, &[bsz, seqlen, dim])
    }

    fn ffn(&mut self, x: &GpuTensor, input_ids: Option<&[u32]>) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = self.config.hidden_size;
        let device = x.device.clone();
        let total = bsz * seqlen;
        let topk = self.config.num_experts_per_tok;
        let n_experts = self.config.n_routed_experts;

        let shared_out = self.ffn_shared(x)?;
        let gate_output = self.gate.forward(x, input_ids)?;

        if self.layer_id == 42 {
            Self::debug_tensor_stats("ffn_shared_out", &shared_out, self.layer_id);
            Self::debug_tensor_stats("ffn_input", x, self.layer_id);
        }

        let mut y_gpu = shared_out;
        let x_flat = self.reshape(x, &[total, dim])?;

        let indices_host = {
            let indices_i32 = gate_output.indices.to_host()?;
            let indices: &[i32] = bytemuck::cast_slice(&indices_i32.data);
            indices.to_vec()
        };

        let mut expert_counts = vec![0usize; n_experts];
        for &idx in &indices_host {
            let e = idx as usize;
            if e < n_experts {
                expert_counts[e] += 1;
            }
        }

        let mut expert_tokens_map: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
        for t in 0..total {
            for k in 0..topk {
                let e = indices_host[t * topk + k] as usize;
                if e < n_experts {
                    expert_tokens_map[e].push((t, k));
                }
            }
        }

        let use_scatter_add = dim == 4096 || dim == 7168;

        for expert_id in 0..n_experts {
            if expert_counts[expert_id] == 0 {
                continue;
            }

            let expert_tokens = &expert_tokens_map[expert_id];
            let n_tokens = expert_tokens.len();
            let row_indices: Vec<usize> = expert_tokens.iter().map(|&(t, _k)| t).collect();

            let x_expert_gpu = if n_tokens <= 32 {
                x_flat.gather_rows(&row_indices, dim)?
            } else {
                let tid_data: Vec<i32> = row_indices.iter().map(|&t| t as i32).collect();
                let tid_cpu = CpuTensor::new(bytemuck::cast_slice(&tid_data).to_vec(), vec![n_tokens], DType::INT32);
                let tid_gpu = GpuTensor::from_host(device.clone(), &tid_cpu)?;
                let x_src = GpuTensor {
                    slice: x_flat.slice.clone(),
                    shape: vec![total, dim],
                    dtype: DType::BF16,
                    device: device.clone(),
                };
                let x_dst = GpuTensor::zeros(device.clone(), vec![n_tokens, dim], DType::BF16)?;
                let kernel_name = format!("moe_gather_D{}", dim);
                self.kernels.call(&kernel_name, &[&x_src, &tid_gpu, &x_dst])
                    .with_context(|| format!("moe_gather kernel {} failed", kernel_name))?;
                x_dst
            };

            let expert_out = self.compute_expert(&x_expert_gpu, expert_id)?;

            if use_scatter_add {
                let weights_host = gate_output.weights.to_host()?;
                let weights_f32: &[f32] = bytemuck::cast_slice(&weights_host.data);

                let mut w_data = vec![0f32; n_tokens];
                let mut tid_data = vec![0i32; n_tokens];
                for (i, &(t, k)) in expert_tokens.iter().enumerate() {
                    w_data[i] = weights_f32[t * topk + k];
                    tid_data[i] = t as i32;
                }
                let w_cpu = CpuTensor::new(bytemuck::cast_slice(&w_data).to_vec(), vec![n_tokens], DType::FP32);
                let tid_cpu = CpuTensor::new(bytemuck::cast_slice(&tid_data).to_vec(), vec![n_tokens], DType::INT32);
                let w_gpu = GpuTensor::from_host(device.clone(), &w_cpu)?;
                let tid_gpu = GpuTensor::from_host(device.clone(), &tid_cpu)?;

                let expert_2d = GpuTensor {
                    slice: expert_out.slice.clone(),
                    shape: vec![n_tokens, dim],
                    dtype: DType::BF16,
                    device: device.clone(),
                };
                let y_2d = GpuTensor {
                    slice: y_gpu.slice.clone(),
                    shape: vec![total, dim],
                    dtype: DType::BF16,
                    device: device.clone(),
                };

                let kernel_name = match dim {
                    4096 => "scatter_add_D4096",
                    7168 => "scatter_add_D7168",
                    _ => "",
                };
                if !kernel_name.is_empty() {
                    match self.kernels.call(kernel_name, &[&expert_2d, &w_gpu, &tid_gpu, &y_2d]) {
                        Ok(()) => {
                            y_gpu = GpuTensor {
                                slice: y_2d.slice,
                                shape: vec![bsz, seqlen, dim],
                                dtype: DType::BF16,
                                device: device.clone(),
                            };
                            continue;
                        }
                        Err(e) => {
                            return Err(e).with_context(|| format!("scatter_add kernel {} failed", kernel_name));
                        }
                    }
                }
            }

            let weights_host = gate_output.weights.to_host()?;
            let weights_f32: &[f32] = bytemuck::cast_slice(&weights_host.data);
            let expert_host = expert_out.to_host()?;
            let expert_bf16: &[half::bf16] = bytemuck::cast_slice(&expert_host.data);
            let y_host = y_gpu.to_host()?;
            let mut y_data: Vec<f32> = {
                let y_bf16: &[half::bf16] = bytemuck::cast_slice(&y_host.data);
                y_bf16.iter().map(|v| v.to_f32()).collect()
            };

            for (i, &(t, k)) in expert_tokens.iter().enumerate() {
                let w = weights_f32[t * topk + k];
                for d in 0..dim {
                    let expert_val = expert_bf16[i * dim + d].to_f32();
                    y_data[t * dim + d] += w * expert_val;
                }
            }

            let y_bf16_out: Vec<u16> = y_data.iter()
                .map(|v| half::bf16::from_f32(*v).to_bits())
                .collect();
            let y_cpu = CpuTensor::new(bytemuck::cast_slice(&y_bf16_out).to_vec(), vec![bsz, seqlen, dim], DType::BF16);
            y_gpu = GpuTensor::from_host(device.clone(), &y_cpu)?;
        }

        let active_expert_ids: Vec<usize> = (0..n_experts)
            .filter(|&e| expert_counts[e] > 0)
            .collect();
        if !active_expert_ids.is_empty() {
            self.expert_scheduler.prefetch_layers_ahead(self.layer_id, &active_expert_ids, &mut self.weight_loader)?;
            self.expert_scheduler.adapt();
        }

        Ok(y_gpu)
    }

    fn compute_expert(&mut self, x: &GpuTensor, expert_id: usize) -> Result<GpuTensor> {
        let dim = self.config.hidden_size;
        let inter_dim = self.config.moe_intermediate_size;
        let device = x.device.clone();

        // 阶段1: GPU 缓存命中 → 直接 GPU GEMM（~0.2ms）
        // 只检查 GPU 缓存，不上传
        if self.expert_scheduler.gpu_cache_contains(self.layer_id, expert_id) {
            if let Ok(gpu_weights) = self.expert_scheduler.get_expert_gpu(self.layer_id, expert_id, true) {
                let m = x.shape.iter().rev().skip(1).product::<usize>();
                let x_flat = GpuTensor {
                    slice: x.slice.clone(),
                    shape: vec![m, dim],
                    dtype: x.dtype,
                    device: device.clone(),
                };

                let result = if self.expert_scheduler.expert_dtype == DType::FP4E2M1 {
                    let gate = self.fp4_gemm_act_quant(&x_flat, &gpu_weights.w1, &gpu_weights.w1_scale, inter_dim)?;
                    let up = self.fp4_gemm_act_quant(&x_flat, &gpu_weights.w3, &gpu_weights.w3_scale, inter_dim)?;
                    let swiglu_out = self.swiglu(&gate, &up)?;
                    let swiglu_flat = GpuTensor {
                        slice: swiglu_out.slice.clone(),
                        shape: vec![m, inter_dim],
                        dtype: swiglu_out.dtype,
                        device: device.clone(),
                    };
                    self.fp4_gemm_act_quant(&swiglu_flat, &gpu_weights.w2, &gpu_weights.w2_scale, dim)?
                } else {
                    let gate = self.fp8_gemm_act_quant(&x_flat, &gpu_weights.w1, &gpu_weights.w1_scale, inter_dim)?;
                    let up = self.fp8_gemm_act_quant(&x_flat, &gpu_weights.w3, &gpu_weights.w3_scale, inter_dim)?;
                    let swiglu_out = self.swiglu(&gate, &up)?;
                    let swiglu_flat = GpuTensor {
                        slice: swiglu_out.slice.clone(),
                        shape: vec![m, inter_dim],
                        dtype: swiglu_out.dtype,
                        device: device.clone(),
                    };
                    self.fp8_gemm_act_quant(&swiglu_flat, &gpu_weights.w2, &gpu_weights.w2_scale, dim)?
                };

                return Ok(result);
            }
        }

        // 阶段2: GPU 缓存未命中 → CPU FFN（~2.7ms，比 DMA 5ms 快）
        // 首先尝试上传到 GPU 并计算（回退到原始路径，保证正确性）
        // TODO: 后续替换为纯 CPU FFN 路径
        self.compute_expert_gpu_upload(x, expert_id)
    }

    /// 原始路径：上传到 GPU 并计算
    fn compute_expert_gpu_upload(&mut self, x: &GpuTensor, expert_id: usize) -> Result<GpuTensor> {
        let dim = self.config.hidden_size;
        let inter_dim = self.config.moe_intermediate_size;
        let device = x.device.clone();

        self.expert_scheduler.ensure_expert_loaded(self.layer_id, expert_id, &mut self.weight_loader)?;

        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let x_flat = GpuTensor {
            slice: x.slice.clone(),
            shape: vec![m, dim],
            dtype: x.dtype,
            device: device.clone(),
        };

        let expert = self.expert_scheduler.upload_expert_raw(self.layer_id, expert_id)?;

        let result = if self.expert_scheduler.expert_dtype == DType::FP4E2M1 {
            let gate = self.fp4_gemm_act_quant(&x_flat, &expert.w1, &expert.w1_scale, inter_dim)?;
            let up = self.fp4_gemm_act_quant(&x_flat, &expert.w3, &expert.w3_scale, inter_dim)?;
            let swiglu_out = self.swiglu(&gate, &up)?;
            let swiglu_flat = GpuTensor {
                slice: swiglu_out.slice.clone(),
                shape: vec![m, inter_dim],
                dtype: swiglu_out.dtype,
                device: device.clone(),
            };
            self.fp4_gemm_act_quant(&swiglu_flat, &expert.w2, &expert.w2_scale, dim)
        } else {
            let gate = self.fp8_gemm_act_quant(&x_flat, &expert.w1, &expert.w1_scale, inter_dim)?;
            let up = self.fp8_gemm_act_quant(&x_flat, &expert.w3, &expert.w3_scale, inter_dim)?;
            let swiglu_out = self.swiglu(&gate, &up)?;
            let swiglu_flat = GpuTensor {
                slice: swiglu_out.slice.clone(),
                shape: vec![m, inter_dim],
                dtype: swiglu_out.dtype,
                device: device.clone(),
            };
            self.fp8_gemm_act_quant(&swiglu_flat, &expert.w2, &expert.w2_scale, dim)
        };

        drop(expert);
        drop(x_flat);

        result
    }

    /// CPU FFN 回退：从 CPU 缓存/SSD 加载权重，在 CPU 上计算 FFN
    fn compute_expert_cpu(&mut self, x: &GpuTensor, expert_id: usize) -> Result<GpuTensor> {
        let dim = self.config.hidden_size;
        let inter_dim = self.config.moe_intermediate_size;
        let device = x.device.clone();

        // 确保 CPU 缓存中有权重
        self.expert_scheduler.ensure_expert_loaded(self.layer_id, expert_id, &mut self.weight_loader)?;

        // 从 GPU 下载 x 到 CPU
        let x_host = x.to_host()?;
        let x_bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let mut x_f32 = vec![0.0f32; m * dim];
        for (i, v) in x_bf16.iter().enumerate() {
            x_f32[i] = v.to_f32();
        }

        // 获取 CPU 缓存中的权重并执行 CPU FFN
        let result_f32 = if self.expert_scheduler.expert_dtype == DType::FP4E2M1 {
            self.cpu_ffn_fp4(&x_f32, expert_id, m, dim, inter_dim)?
        } else {
            // FP8: 反量化到 f32 后用标量 FFN
            self.cpu_ffn_fp8(&x_f32, expert_id, m, dim, inter_dim)?
        };

        // 结果上传回 GPU
        let result_bf16: Vec<u16> = result_f32.iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect();
        let result_cpu = CpuTensor::new(
            bytemuck::cast_slice(&result_bf16).to_vec(),
            vec![m, dim],
            DType::BF16,
        );
        GpuTensor::from_host(device, &result_cpu)
    }

    /// FP4 CPU FFN：使用 AVX-512 优化的 cpu_expert 模块
    fn cpu_ffn_fp4(
        &mut self,
        x: &[f32],
        expert_id: usize,
        m: usize,
        dim: usize,
        inter_dim: usize,
    ) -> Result<Vec<f32>> {
        use crate::cpu_expert::kernel::{Fp4Weight, fp4_expert_ffn_pair_amd7600};

        let cpu_weights = self.expert_scheduler.get_cpu_weights(self.layer_id, expert_id)
            .ok_or_else(|| anyhow!("expert {}/{} not in CPU cache", self.layer_id, expert_id))?;

        // safetensors 中 FP4 权重 shape=[out_dim, in_dim/2]（packed），scale shape=[out_dim, in_dim/32]
        // Fp4Weight 期望逻辑 shape=(out_dim, in_dim)
        let gate_w = Fp4Weight::new(
            cpu_weights.w1.data.clone(),
            cpu_weights.w1_scale.data.clone(),
            (inter_dim, dim),
        );
        let up_w = Fp4Weight::new(
            cpu_weights.w3.data.clone(),
            cpu_weights.w3_scale.data.clone(),
            (inter_dim, dim),
        );
        let down_w = Fp4Weight::new(
            cpu_weights.w2.data.clone(),
            cpu_weights.w2_scale.data.clone(),
            (dim, inter_dim),
        );

        // 逐 token 调用 CPU FFN
        let mut result = vec![0.0f32; m * dim];
        for t in 0..m {
            let x_t = &x[t * dim..(t + 1) * dim];
            let out_t = fp4_expert_ffn_pair_amd7600(x_t, &gate_w, &up_w, &down_w, 1.0, 0.0);
            result[t * dim..(t + 1) * dim].copy_from_slice(&out_t);
        }

        Ok(result)
    }

    /// FP8 CPU FFN：反量化到 f32 后标量计算
    fn cpu_ffn_fp8(
        &mut self,
        x: &[f32],
        expert_id: usize,
        _m: usize,
        dim: usize,
        inter_dim: usize,
    ) -> Result<Vec<f32>> {
        let cpu_weights = self.expert_scheduler.get_cpu_weights(self.layer_id, expert_id)
            .ok_or_else(|| anyhow!("expert {}/{} not in CPU cache", self.layer_id, expert_id))?;

        // 反量化 FP8 权重到 f32
        let w1_f32 = quant::dequant_fp8_e4m3_to_f32(&cpu_weights.w1.data, &cpu_weights.w1_scale.data, &cpu_weights.w1.shape)?;
        let w3_f32 = quant::dequant_fp8_e4m3_to_f32(&cpu_weights.w3.data, &cpu_weights.w3_scale.data, &cpu_weights.w3.shape)?;
        let w2_f32 = quant::dequant_fp8_e4m3_to_f32(&cpu_weights.w2.data, &cpu_weights.w2_scale.data, &cpu_weights.w2.shape)?;

        // 标量 matvec
        let mut result = Vec::new();
        for t in 0..x.len() / dim {
            let x_t = &x[t * dim..(t + 1) * dim];
            // gate
            let gate = matvec_f32(&w1_f32, x_t, inter_dim, dim);
            // up
            let up = matvec_f32(&w3_f32, x_t, inter_dim, dim);
            // SwiGLU
            let mut mid = vec![0.0f32; inter_dim];
            for i in 0..inter_dim {
                let g = gate[i];
                let u = up[i];
                let sig = 1.0 / (1.0 + (-g).exp());
                mid[i] = g * sig * u;
            }
            // down
            let out = matvec_f32(&w2_f32, &mid, dim, inter_dim);
            result.extend_from_slice(&out);
        }

        Ok(result)
    }

    fn act_quant_inplace_nope(&self, kv: &GpuTensor) -> Result<GpuTensor> {
        let device = kv.device.clone();
        let shape = kv.shape.clone();
        let head_dim = self.config.head_dim;
        let rope_dim = self.config.qk_rope_head_dim;
        let nope_dim = head_dim - rope_dim;
        let block_size = 64usize;

        if nope_dim == 448 && block_size == 64 {
            let kernel_name = "act_quant_N448_bs64_inplace";
            let n_rows: usize = shape.iter().rev().skip(1).product();

            let nope_gpu = self.slice_columns(kv, 0, nope_dim)?;
            let nope_2d = GpuTensor {
                slice: nope_gpu.slice,
                shape: vec![n_rows, nope_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let y_nope = GpuTensor::zeros(device.clone(), vec![n_rows, nope_dim], DType::BF16)?;
            let n_blocks = nope_dim / block_size;
            let s = GpuTensor::zeros(device.clone(), vec![n_rows, n_blocks], DType::FP32)?;

            self.kernels.call(kernel_name, &[&nope_2d, &y_nope, &s])
                .with_context(|| "act_quant_N448_bs64_inplace kernel failed")?;

            let rope_gpu = self.slice_columns(kv, nope_dim, head_dim)?;
            return self.concat_columns(&y_nope, &rope_gpu, &shape);
        }

        Err(anyhow!(
            "act_quant_inplace_nope: no GPU kernel for nope_dim={}, block_size={}",
            nope_dim, block_size
        ))
    }

    fn rmsnorm(&self, x: &GpuTensor, weight: Option<&GpuTensor>) -> Result<GpuTensor> {
        if x.dtype != DType::BF16 {
            return Err(anyhow!("rmsnorm: x.dtype must be BF16, got {:?}", x.dtype));
        }
        let device = x.device.clone();
        let shape = x.shape.clone();
        let last_dim = shape.last().copied().unwrap_or(1);

        if let Some(w) = weight {
            if w.dtype != DType::BF16 && w.dtype != DType::FP32 {
                return Err(anyhow!("rmsnorm: weight.dtype must be BF16 or FP32, got {:?}", w.dtype));
            }
            let w_len: usize = w.shape.iter().product();
            if w_len != last_dim {
                return Err(anyhow!("rmsnorm: weight length {} != last_dim {}", w_len, last_dim));
            }
        }

        let kernel_name = if weight.is_some() {
            match last_dim {
                4096 => Some("rmsnorm_N4096"),
                1024 => Some("rmsnorm_N1024"),
                512 => Some("rmsnorm_N512"),
                _ => None,
            }
        } else {
            match last_dim {
                1024 => Some("rmsnorm_no_weight_N1024"),
                512 => Some("rmsnorm_no_weight_N512"),
                _ => None,
            }
        };

        if let Some(name) = kernel_name {
            let n_rows: usize = shape.iter().rev().skip(1).product();
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![n_rows, last_dim],
                dtype: x.dtype,
                device: device.clone(),
            };

            let w_tensor = if let Some(w) = weight {
                w.clone()
            } else {
                GpuTensor::zeros(device.clone(), vec![last_dim], DType::FP32)?
            };
            let w_f32 = self.cast_to_f32(&w_tensor)?;

            let y = GpuTensor::zeros(device.clone(), vec![n_rows, last_dim], DType::BF16)?;

            self.kernels.call(name, &[&x_2d, &w_f32, &y])
                .with_context(|| format!("rmsnorm kernel {} failed", name))?;

            return Ok(GpuTensor {
                slice: y.slice,
                shape,
                dtype: DType::BF16,
                device,
            });
        }

        Err(anyhow!(
            "rmsnorm: no GPU kernel for last_dim={}, has_weight={}",
            last_dim, weight.is_some()
        ))
    }

    fn rmsnorm_f32(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype != DType::FP32 {
            return Err(anyhow!("rmsnorm_f32: x.dtype must be FP32, got {:?}", x.dtype));
        }
        let device = x.device.clone();
        let shape = x.shape.clone();
        let last_dim = shape.last().copied().unwrap_or(1);
        let n = shape.iter().product::<usize>();
        let m = n / last_dim;

        let kernel_name = match last_dim {
            4096 => Some("rmsnorm_f32_N4096"),
            7168 => Some("rmsnorm_f32_N7168"),
            16384 => Some("rmsnorm_f32_N16384"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::FP32,
                device: device.clone(),
            };
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::FP32)?;
            self.kernels.call(kname, &[&x_2d, &y_2d])
                .with_context(|| format!("rmsnorm_f32 kernel {} failed", kname))?;
            return Ok(GpuTensor {
                slice: y_2d.slice,
                shape,
                dtype: DType::FP32,
                device,
            });
        }

        Err(anyhow!(
            "rmsnorm_f32: no GPU kernel for last_dim={}",
            last_dim
        ))
    }

    fn rmsnorm_f32_rsqrt_only(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype != DType::FP32 {
            return Err(anyhow!("rmsnorm_f32_rsqrt_only: x.dtype must be FP32, got {:?}", x.dtype));
        }
        let device = x.device.clone();
        let shape = x.shape.clone();
        let last_dim = shape.last().copied().unwrap_or(1);
        let n = shape.iter().product::<usize>();
        let m = n / last_dim;

        let kernel_name = match last_dim {
            4096 => Some("rmsnorm_rsqrt_f32_N4096"),
            7168 => Some("rmsnorm_rsqrt_f32_N7168"),
            16384 => Some("rmsnorm_rsqrt_f32_N16384"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::FP32,
                device: device.clone(),
            };
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, 1], DType::FP32)?;
            self.kernels.call(kname, &[&x_2d, &y_2d])
                .with_context(|| format!("rmsnorm_rsqrt kernel {} failed", kname))?;
            return Ok(GpuTensor {
                slice: y_2d.slice,
                shape: vec![m, 1],
                dtype: DType::FP32,
                device,
            });
        }

        Err(anyhow!(
            "rmsnorm_f32_rsqrt_only: no GPU kernel for last_dim={}",
            last_dim
        ))
    }

    fn mul_row_broadcast(&self, mat: &GpuTensor, row_vec: &GpuTensor) -> Result<GpuTensor> {
        if mat.dtype != DType::FP32 || row_vec.dtype != DType::FP32 {
            return Err(anyhow!("mul_row_broadcast: both must be FP32, got mat={:?} vec={:?}", mat.dtype, row_vec.dtype));
        }
        if mat.shape.len() != 2 {
            return Err(anyhow!("mul_row_broadcast: mat must be 2D, got {:?}", mat.shape));
        }
        if row_vec.shape.len() != 2 || row_vec.shape[1] != 1 {
            return Err(anyhow!("mul_row_broadcast: row_vec must be [m, 1], got {:?}", row_vec.shape));
        }
        if mat.shape[0] != row_vec.shape[0] {
            return Err(anyhow!("mul_row_broadcast: row count mismatch mat={} vec={}", mat.shape[0], row_vec.shape[0]));
        }

        let m = mat.shape[0];
        let n = mat.shape[1];
        let device = mat.device.clone();

        let out = GpuTensor::zeros(device.clone(), vec![m, n], DType::FP32)?;

        self.kernels.call(
            "mul_row_broadcast_f32",
            &[mat, row_vec, &out],
        ).with_context(|| "mul_row_broadcast_f32 kernel failed")?;

        Ok(out)
    }

    fn fp8_gemm_act_quant(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        weight_scale: &GpuTensor,
        out_dim: usize,
    ) -> Result<GpuTensor> {
        if x.dtype != DType::BF16 {
            return Err(anyhow!("fp8_gemm_act_quant: x.dtype must be BF16, got {:?}", x.dtype));
        }
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        let n = out_dim;

        if weight.dtype != DType::FP8E4M3 {
            return Err(anyhow!("fp8_gemm_act_quant: weight.dtype must be FP8E4M3, got {:?}", weight.dtype));
        }
        if weight.shape.len() != 2 || weight.shape[0] != n || weight.shape[1] != k {
            return Err(anyhow!(
                "fp8_gemm_act_quant: weight shape must be [{}, {}], got {:?}",
                n, k, weight.shape
            ));
        }
        let expected_ws_rows = (n + 127) / 128;
        let expected_ws_cols = (k + 127) / 128;
        if weight_scale.shape.len() != 2 || weight_scale.shape[0] != expected_ws_rows || weight_scale.shape[1] != expected_ws_cols {
            return Err(anyhow!(
                "fp8_gemm_act_quant: weight_scale shape must be [{}, {}], got {:?}",
                expected_ws_rows, expected_ws_cols, weight_scale.shape
            ));
        }
        if weight_scale.dtype != DType::FP8E8M0 {
            return Err(anyhow!("fp8_gemm_act_quant: weight_scale.dtype must be FP8E8M0, got {:?}", weight_scale.dtype));
        }

        let device = x.device.clone();

        let kernel_name = match (n, k) {
            (32768, 1024) => Some("fp8_gemm_N32768_K1024"),
            (512, 4096) => Some("fp8_gemm_N512_K4096"),
            (1024, 4096) => Some("fp8_gemm_N1024_K4096"),
            (4096, 8192) => Some("fp8_gemm_N4096_K8192"),
            (2048, 4096) => Some("fp8_gemm_N2048_K4096"),
            (4096, 2048) => Some("fp8_gemm_N4096_K2048"),
            (8192, 1024) => Some("fp8_gemm_N8192_K1024"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let (x_fp8, x_scale) = self.act_quant_gpu(x, 128)?;

            let x_2d = GpuTensor {
                slice: x_fp8.slice,
                shape: vec![m, k],
                dtype: DType::FP8E4M3,
                device: device.clone(),
            };
            let x_s_2d = GpuTensor {
                slice: x_scale.slice,
                shape: vec![m, k / 128],
                dtype: x_scale.dtype,
                device: device.clone(),
            };

            let w_2d = GpuTensor {
                slice: weight.slice.clone(),
                shape: vec![n, k],
                dtype: DType::FP8E4M3,
                device: device.clone(),
            };
            let w_s_2d = GpuTensor {
                slice: weight_scale.slice.clone(),
                shape: weight_scale.shape.clone(),
                dtype: weight_scale.dtype,
                device: device.clone(),
            };

            let c = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16)?;

            self.kernels.call(kname, &[&x_2d, &w_2d, &c, &x_s_2d, &w_s_2d])
                .with_context(|| format!("fp8_gemm kernel N={} K={} failed", n, k))?;
            return Ok(c);
        }

        Err(anyhow!(
            "fp8_gemm_act_quant: no GPU kernel for (N={}, K={}, weight_dtype={})",
            n, k, weight.dtype
        ))
    }

    fn fp4_gemm_act_quant(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        weight_scale: &GpuTensor,
        out_dim: usize,
    ) -> Result<GpuTensor> {
        if x.dtype != DType::BF16 {
            return Err(anyhow!("fp4_gemm_act_quant: x.dtype must be BF16, got {:?}", x.dtype));
        }
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        let n = out_dim;

        if weight.dtype != DType::FP4E2M1 {
            return Err(anyhow!("fp4_gemm_act_quant: weight.dtype must be FP4E2M1, got {:?}", weight.dtype));
        }
        if weight.shape.len() != 2 || weight.shape[0] != n || weight.shape[1] != k / 2 {
            return Err(anyhow!(
                "fp4_gemm_act_quant: weight shape must be [{}, {}], got {:?}",
                n, k / 2, weight.shape
            ));
        }
        let expected_ws_cols = (k + 31) / 32;
        if weight_scale.shape.len() != 2 || weight_scale.shape[0] != n || weight_scale.shape[1] != expected_ws_cols {
            return Err(anyhow!(
                "fp4_gemm_act_quant: weight_scale shape must be [{}, {}], got {:?}",
                n, expected_ws_cols, weight_scale.shape
            ));
        }
        if weight_scale.dtype != DType::FP8E8M0 {
            return Err(anyhow!("fp4_gemm_act_quant: weight_scale.dtype must be FP8E8M0, got {:?}", weight_scale.dtype));
        }

        let device = x.device.clone();

        let kernel_name = match (n, k) {
            (2048, 4096) => Some("fp4_gemm_N2048_K4096"),
            (4096, 2048) => Some("fp4_gemm_N4096_K2048"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let (x_fp8, x_scale) = self.act_quant_gpu(x, 128)?;

            let x_2d = GpuTensor {
                slice: x_fp8.slice,
                shape: vec![m, k],
                dtype: DType::FP8E4M3,
                device: device.clone(),
            };
            let x_s_2d = GpuTensor {
                slice: x_scale.slice,
                shape: vec![m, k / 128],
                dtype: x_scale.dtype,
                device: device.clone(),
            };

            let w_2d = GpuTensor {
                slice: weight.slice.clone(),
                shape: vec![n, k / 2],
                dtype: DType::FP4E2M1,
                device: device.clone(),
            };
            let w_s_2d = GpuTensor {
                slice: weight_scale.slice.clone(),
                shape: weight_scale.shape.clone(),
                dtype: weight_scale.dtype,
                device: device.clone(),
            };

            let c = GpuTensor::zeros(device.clone(), vec![m, n], DType::BF16)?;

            self.kernels.call(kname, &[&x_2d, &w_2d, &c, &x_s_2d, &w_s_2d])
                .with_context(|| format!("fp4_gemm kernel N={} K={} failed", n, k))?;
            return Ok(c);
        }

        Err(anyhow!(
            "fp4_gemm_act_quant: no GPU kernel for (N={}, K={}, weight_dtype={})",
            n, k, weight.dtype
        ))
    }

    fn act_quant_gpu(&self, x: &GpuTensor, block_size: usize) -> Result<(GpuTensor, GpuTensor)> {
        if x.dtype != DType::BF16 {
            return Err(anyhow!("act_quant_gpu: x.dtype must be BF16, got {:?}", x.dtype));
        }
        let device = x.device.clone();
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        if k % block_size != 0 {
            return Err(anyhow!("act_quant_gpu: K={} must be divisible by block_size={}", k, block_size));
        }
        let n_blocks = k / block_size;

        let kernel_name = match (k, block_size) {
            (4096, 128) => Some("act_quant_N4096_bs128"),
            (8192, 128) => Some("act_quant_N8192_bs128"),
            (2048, 128) => Some("act_quant_N2048_bs128"),
            (1024, 128) => Some("act_quant_N1024_bs128"),
            _ => None,
        };

        if let Some(name) = kernel_name {
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, k],
                dtype: x.dtype,
                device: device.clone(),
            };
            let y = GpuTensor::zeros(device.clone(), vec![m, k], DType::FP8E4M3)?;
            let s = GpuTensor::zeros(device.clone(), vec![m, n_blocks], DType::FP8E8M0)?;

            self.kernels.call(name, &[&x_2d, &y, &s])
                .with_context(|| format!("act_quant kernel {} failed", name))?;
            return Ok((y, s));
        }

        Err(anyhow!(
            "act_quant_gpu: no GPU kernel for (K={}, block_size={})",
            k, block_size
        ))
    }

    fn apply_rope_q(&self, q: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = q.device.clone();
        let shape = q.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let n_heads = shape[2];
        let head_dim = shape[3];
        let rope_dim = self.config.qk_rope_head_dim;
        let nope_dim = head_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let q_3d = GpuTensor {
                slice: q.slice.clone(),
                shape: vec![total, n_heads, head_dim],
                dtype: q.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&q_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&q_3d, nope_dim, head_dim)?;

            let (cos_gpu, sin_gpu) = self.get_rope_freqs(rope, start_pos, seqlen, total, &device)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, n_heads, rope_dim], DType::BF16)?;

            self.kernels.call("rope_interleaved_fwd_D64", &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope])
                .with_context(|| "rope_interleaved_fwd_D64 kernel failed")?;

            let out_shape = vec![total, n_heads, head_dim];
            let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
            return Ok(GpuTensor {
                slice: result.slice,
                shape,
                dtype: DType::BF16,
                device,
            });
        }

        Err(anyhow!("apply_rope_q: no GPU kernel for rope_dim={}", rope_dim))
    }

    fn apply_rope_kv(&self, kv: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = kv.device.clone();
        let shape = kv.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let kv_dim = shape[2];
        let rope_dim = self.config.qk_rope_head_dim;
        let nope_dim = kv_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let kv_3d = GpuTensor {
                slice: kv.slice.clone(),
                shape: vec![total, 1, kv_dim],
                dtype: kv.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&kv_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&kv_3d, nope_dim, kv_dim)?;

            let (cos_gpu, sin_gpu) = self.get_rope_freqs(rope, start_pos, seqlen, total, &device)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, 1, rope_dim], DType::BF16)?;

            self.kernels.call("rope_interleaved_fwd_D64", &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope])
                .with_context(|| "rope_interleaved_fwd_D64 kernel for KV failed")?;

            let out_shape = vec![total, 1, kv_dim];
            let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
            return Ok(GpuTensor {
                slice: result.slice,
                shape,
                dtype: DType::BF16,
                device,
            });
        }

        Err(anyhow!("apply_rope_kv: no GPU kernel for rope_dim={}", rope_dim))
    }

    fn apply_inverse_rope(&self, o: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = o.device.clone();
        let shape = o.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let n_heads = shape[2];
        let head_dim = shape[3];
        let rope_dim = self.config.qk_rope_head_dim;
        let nope_dim = head_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let o_3d = GpuTensor {
                slice: o.slice.clone(),
                shape: vec![total, n_heads, head_dim],
                dtype: o.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&o_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&o_3d, nope_dim, head_dim)?;

            let (cos_gpu, sin_gpu) = self.get_rope_freqs(rope, start_pos, seqlen, total, &device)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, n_heads, rope_dim], DType::BF16)?;

            self.kernels.call("rope_interleaved_inv_D64", &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope])
                .with_context(|| "rope_interleaved_inv_D64 kernel failed")?;

            let out_shape = vec![total, n_heads, head_dim];
            let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
            return Ok(GpuTensor {
                slice: result.slice,
                shape,
                dtype: DType::BF16,
                device,
            });
        }

        Err(anyhow!("apply_inverse_rope: no GPU kernel for rope_dim={}", rope_dim))
    }

    fn get_rope_freqs(
        &self,
        rope: &RopeCache,
        start_pos: usize,
        seqlen: usize,
        total: usize,
        device: &Arc<CudaContext>,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let half_rope = self.config.qk_rope_head_dim / 2;
        if let Some((c, s)) = rope.get_gpu_slice(start_pos, seqlen) {
            Ok((c, s))
        } else {
            let (cos_data, sin_data) = rope.get_slice(start_pos, seqlen);
            let mut cos_expanded = vec![0f32; total * half_rope];
            let mut sin_expanded = vec![0f32; total * half_rope];
            for s in 0..seqlen {
                for k in 0..half_rope {
                    cos_expanded[s * half_rope + k] = cos_data[s * half_rope + k];
                    sin_expanded[s * half_rope + k] = sin_data[s * half_rope + k];
                }
            }
            let cos_cpu = CpuTensor::new(bytemuck::cast_slice(&cos_expanded).to_vec(), vec![total, half_rope], DType::FP32);
            let sin_cpu = CpuTensor::new(bytemuck::cast_slice(&sin_expanded).to_vec(), vec![total, half_rope], DType::FP32);
            Ok((
                GpuTensor::from_host(device.clone(), &cos_cpu)?,
                GpuTensor::from_host(device.clone(), &sin_cpu)?,
            ))
        }
    }

    fn sparse_attention(
        &self,
        q: &GpuTensor,
        kv: &GpuTensor,
        attn_sink: &GpuTensor,
        topk_idxs: &GpuTensor,
        bsz: usize,
        seqlen: usize,
        _start_pos: usize,
    ) -> Result<GpuTensor> {
        let n_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let device = q.device.clone();

        if q.shape.len() != 4 || q.shape[0] != bsz || q.shape[1] != seqlen || q.shape[2] != n_heads || q.shape[3] != head_dim {
            return Err(anyhow!(
                "sparse_attention: q shape must be [{}, {}, {}, {}], got {:?}",
                bsz, seqlen, n_heads, head_dim, q.shape
            ));
        }
        if kv.shape.len() != 3 || kv.shape[2] != head_dim {
            return Err(anyhow!(
                "sparse_attention: kv shape must be [*, *, {}], got {:?}",
                head_dim, kv.shape
            ));
        }
        if attn_sink.dtype != DType::FP32 || attn_sink.shape.iter().product::<usize>() != n_heads {
            return Err(anyhow!(
                "sparse_attention: attn_sink must be FP32 with {} elements, got {:?}{:?}",
                n_heads, attn_sink.dtype, attn_sink.shape
            ));
        }
        if topk_idxs.dtype != DType::INT32 {
            return Err(anyhow!("sparse_attention: topk_idxs must be INT32, got {:?}", topk_idxs.dtype));
        }

        if n_heads == 64 && head_dim == 512 {
            let sink_f32 = self.cast_to_f32(attn_sink)?;
            let sink_1d = GpuTensor {
                slice: sink_f32.slice,
                shape: vec![n_heads],
                dtype: DType::FP32,
                device: device.clone(),
            };

            let o = GpuTensor::zeros(
                device.clone(),
                vec![bsz, seqlen, n_heads, head_dim],
                DType::BF16,
            )?;

            self.kernels.call("sparse_attn_h64_d512", &[q, kv, &o, &sink_1d, topk_idxs])
                .with_context(|| "sparse_attn_h64_d512 kernel failed")?;
            return Ok(o);
        }

        Err(anyhow!(
            "sparse_attention: no GPU kernel for n_heads={}, head_dim={}",
            n_heads, head_dim
        ))
    }

    fn compute_topk_idxs(
        &mut self,
        bsz: usize,
        seqlen: usize,
        start_pos: usize,
    ) -> Result<GpuTensor> {
        let win = self.config.sliding_window;
        let device = self.kv_cache.cache.device.clone();

        if start_pos >= win - 1 && bsz == 1 && seqlen == 1 {
            if self.decode_topk_cache.is_none() {
                let mut data = vec![0i32; bsz * seqlen * win];
                for b in 0..bsz {
                    for s in 0..seqlen {
                        let base = (b * seqlen + s) * win;
                        for i in 0..win {
                            data[base + i] = if i < win - 1 {
                                (1 + i) as i32
                            } else {
                                (i - (win - 1)) as i32
                            };
                        }
                    }
                }
                let cpu = CpuTensor::new(
                    bytemuck::cast_slice(&data).to_vec(),
                    vec![bsz, seqlen, win],
                    DType::INT32,
                );
                self.decode_topk_cache = Some(GpuTensor::from_host(device.clone(), &cpu)?);
            }
            return Ok(self.decode_topk_cache.clone().unwrap());
        }

        let idx_data = if start_pos >= win - 1 {
            let sp = start_pos % win;
            let mut data = vec![0i32; bsz * seqlen * win];
            for b in 0..bsz {
                for s in 0..seqlen {
                    let base = (b * seqlen + s) * win;
                    for i in 0..win {
                        data[base + i] = if i < win - sp - 1 {
                            (sp + 1 + i) as i32
                        } else {
                            (i - (win - sp - 1)) as i32
                        };
                    }
                }
            }
            (data, vec![bsz, seqlen, win])
        } else if start_pos > 0 {
            let count = start_pos + 1;
            let mut data = vec![-1i32; bsz * seqlen * win];
            for b in 0..bsz {
                for s in 0..seqlen {
                    let base = (b * seqlen + s) * win;
                    for i in 0..count.min(win) {
                        data[base + i] = i as i32;
                    }
                }
            }
            (data, vec![bsz, seqlen, win])
        } else {
            let count = seqlen.min(win);
            let mut data = vec![-1i32; bsz * seqlen * count];
            for b in 0..bsz {
                for s in 0..seqlen {
                    let base = (b * seqlen + s) * count;
                    for t in 0..count {
                        let pos = (s as i64 - win as i64 + 1).max(0) as usize + t;
                        if pos <= s {
                            data[base + t] = pos as i32;
                        }
                    }
                }
            }
            (data, vec![bsz, seqlen, count])
        };

        let (idx_data, idx_shape) = idx_data;
        let cpu = CpuTensor::new(
            bytemuck::cast_slice(&idx_data).to_vec(),
            idx_shape,
            DType::INT32,
        );
        GpuTensor::from_host(device, &cpu)
    }

    fn get_compress_topk_uniform(
        &self,
        ratio: usize,
        bsz: usize,
        seqlen: usize,
        start_pos: usize,
        offset: usize,
    ) -> Result<GpuTensor> {
        let end_pos = start_pos + seqlen;
        let n_comp = if start_pos > 0 { (start_pos + 1) / ratio } else { end_pos / ratio };
        let device = self.kv_cache.cache.device.clone();

        let mut idx_data = vec![0i32; bsz * seqlen * n_comp];
        for b in 0..bsz {
            for s in 0..seqlen {
                for t in 0..n_comp {
                    if start_pos == 0 {
                        if t >= (s + 1) / ratio {
                            idx_data[(b * seqlen + s) * n_comp + t] = -1;
                        } else {
                            idx_data[(b * seqlen + s) * n_comp + t] = (t + offset) as i32;
                        }
                    } else {
                        idx_data[(b * seqlen + s) * n_comp + t] = (t + offset) as i32;
                    }
                }
            }
        }

        let cpu = CpuTensor::new(
            bytemuck::cast_slice(&idx_data).to_vec(),
            vec![bsz, seqlen, n_comp],
            DType::INT32,
        );
        GpuTensor::from_host(device, &cpu)
    }

    fn concat_topk(
        &self,
        window_topk: &GpuTensor,
        compress_topk: &GpuTensor,
        bsz: usize,
        seqlen: usize,
    ) -> Result<GpuTensor> {
        let device = window_topk.device.clone();
        let w_topk = window_topk.shape[2];
        let c_topk = compress_topk.shape[2];
        let total_topk = w_topk + c_topk;
        let elem_size = 4usize;
        let w_row_bytes = w_topk * elem_size;
        let c_row_bytes = c_topk * elem_size;
        let total_row_bytes = total_topk * elem_size;
        let n_rows = bsz * seqlen;

        let out = GpuTensor::zeros(device.clone(), vec![bsz, seqlen, total_topk], DType::INT32)?;

        let stream = device.default_stream();
        {
            let (w_ptr, _w_guard) = window_topk.slice.device_ptr(&stream);
            let (c_ptr, _c_guard) = compress_topk.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            unsafe {
                for r in 0..n_rows {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * total_row_bytes) as u64,
                        w_ptr + (r * w_row_bytes) as u64,
                        w_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * total_row_bytes + w_row_bytes) as u64,
                        c_ptr + (r * c_row_bytes) as u64,
                        c_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            stream.synchronize()?;
        }

        Ok(out)
    }

    fn concat_kv(
        &self,
        window_kv: &GpuTensor,
        compressed_kv: &GpuTensor,
        bsz: usize,
        _seqlen: usize,
    ) -> Result<GpuTensor> {
        let device = window_kv.device.clone();
        let w_len = window_kv.shape[1];
        let c_len = compressed_kv.shape[1];
        let head_dim = self.config.head_dim;
        let total_len = w_len + c_len;
        let elem_size = window_kv.dtype.element_size();
        let w_row_bytes = w_len * head_dim * elem_size;
        let c_row_bytes = c_len * head_dim * elem_size;
        let total_row_bytes = total_len * head_dim * elem_size;

        let out = GpuTensor::zeros(device.clone(), vec![bsz, total_len, head_dim], window_kv.dtype)?;

        let stream = device.default_stream();
        {
            let (w_ptr, _w_guard) = window_kv.slice.device_ptr(&stream);
            let (c_ptr, _c_guard) = compressed_kv.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            unsafe {
                for b in 0..bsz {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (b * total_row_bytes) as u64,
                        w_ptr + (b * w_row_bytes) as u64,
                        w_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (b * total_row_bytes + w_row_bytes) as u64,
                        c_ptr + (b * c_row_bytes) as u64,
                        c_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            stream.synchronize()?;
        }

        Ok(out)
    }

    fn swiglu(&self, gate: &GpuTensor, up: &GpuTensor) -> Result<GpuTensor> {
        let device = gate.device.clone();
        let shape = gate.shape.clone();
        let last_dim = shape.last().copied().unwrap_or(1);

        if gate.shape != up.shape {
            return Err(anyhow!("swiglu: gate shape {:?} != up shape {:?}", gate.shape, up.shape));
        }
        if gate.dtype != DType::BF16 || up.dtype != DType::BF16 {
            return Err(anyhow!("swiglu: inputs must be BF16, got gate={:?} up={:?}", gate.dtype, up.dtype));
        }

        if last_dim == 2048 {
            let y = GpuTensor::zeros(device.clone(), shape.clone(), DType::BF16)?;
            self.kernels.call("swiglu_N2048", &[gate, up, &y])
                .with_context(|| "swiglu_N2048 kernel failed")?;
            return Ok(y);
        }

        Err(anyhow!(
            "swiglu: no GPU kernel for last_dim={}",
            last_dim
        ))
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
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            128 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N128"),
            512 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N512"),
            1024 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N1024"),
            4096 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N4096"),
            8192 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N8192"),
            16384 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N16384"),
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

        Err(anyhow!(
            "cast_to_f32: no GPU kernel for last_dim={}, dtype={}",
            last_dim, x.dtype
        ))
    }

    fn cast_to_bf16(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype == DType::BF16 {
            return Ok(x.clone());
        }
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            128 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N128"),
            512 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N512"),
            1024 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N1024"),
            4096 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N4096"),
            16384 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N16384"),
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

        Err(anyhow!(
            "cast_to_bf16: no GPU kernel for last_dim={}, dtype={}",
            last_dim, x.dtype
        ))
    }

    fn slice_columns(&self, x: &GpuTensor, col_start: usize, col_end: usize) -> Result<GpuTensor> {
        let device = x.device.clone();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let n_cols = col_end - col_start;
        let n_rows: usize = x.shape.iter().rev().skip(1).product();
        let elem_size = x.dtype.element_size();
        let src_row_bytes = last_dim * elem_size;
        let dst_row_bytes = n_cols * elem_size;
        let src_col_offset = col_start * elem_size;

        let out = GpuTensor::zeros(device.clone(), {
            let mut s = x.shape.clone();
            *s.last_mut().unwrap() = n_cols;
            s
        }, x.dtype)?;

        let stream = device.default_stream();
        {
            let (src_ptr, _src_guard) = x.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            unsafe {
                for r in 0..n_rows {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * dst_row_bytes) as u64,
                        src_ptr + (r * src_row_bytes + src_col_offset) as u64,
                        dst_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            stream.synchronize()?;
        }

        Ok(out)
    }

    fn concat_columns(&self, left: &GpuTensor, right: &GpuTensor, out_shape: &[usize]) -> Result<GpuTensor> {
        let device = left.device.clone();
        let left_cols = *left.shape.last().unwrap_or(&1);
        let right_cols = *right.shape.last().unwrap_or(&1);
        let n_rows: usize = left.shape.iter().rev().skip(1).product();
        let elem_size = left.dtype.element_size();
        let left_row_bytes = left_cols * elem_size;
        let right_row_bytes = right_cols * elem_size;
        let total_row_bytes = (left_cols + right_cols) * elem_size;

        let out = GpuTensor::zeros(device.clone(), out_shape.to_vec(), left.dtype)?;

        let stream = device.default_stream();
        {
            let (left_ptr, _left_guard) = left.slice.device_ptr(&stream);
            let (right_ptr, _right_guard) = right.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);

            unsafe {
                for r in 0..n_rows {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * total_row_bytes) as u64,
                        left_ptr + (r * left_row_bytes) as u64,
                        left_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * total_row_bytes + left_row_bytes) as u64,
                        right_ptr + (r * right_row_bytes) as u64,
                        right_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            stream.synchronize()?;
        }

        Ok(out)
    }
}
