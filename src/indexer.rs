use crate::compressor::Compressor;
use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;

pub struct Indexer {
    pub wq_b: GpuTensor,
    pub weights_proj: GpuTensor,
    pub compressor: Compressor,
    pub kv_cache: GpuTensor,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub index_topk: usize,
    pub compress_ratio: usize,
    pub softmax_scale: f32,
    pub device: Arc<CudaContext>,
    pub cublas: Arc<CublasHandle>,
    kv_cache_cpu: Vec<half::bf16>,
    kernels: Arc<KernelRegistry>,
}

impl Indexer {
    pub fn new(
        device: Arc<CudaContext>,
        config: &ModelConfig,
        layer_id: usize,
        compress_ratio: usize,
        max_batch: usize,
        max_seqlen: usize,
        loader: &mut WeightLoader,
        cublas: Arc<CublasHandle>,
        kernels: Arc<KernelRegistry>,
    ) -> Result<Self> {
        let p = format!("layers.{}.attn.indexer.", layer_id);

        let wq_b = if loader.contains(&(p.clone() + "wq_b.weight")) {
            let cpu = loader.load(&(p.clone() + "wq_b.weight"))?;
            GpuTensor::from_host(device.clone(), &cpu)?
        } else {
            anyhow::bail!("indexer wq_b not found for layer {}", layer_id);
        };

        let weights_proj = if loader.contains(&(p.clone() + "weights_proj.weight")) {
            let cpu = loader.load(&(p.clone() + "weights_proj.weight"))?;
            GpuTensor::from_host(device.clone(), &cpu)?
        } else {
            let n_heads = config.index_n_heads;
            let dim = config.hidden_size;
            GpuTensor::zeros(device.clone(), vec![n_heads, dim], DType::BF16)?
        };

        let n_heads = config.index_n_heads;
        let head_dim = config.index_head_dim;
        let rope_head_dim = config.qk_rope_head_dim;
        let index_topk = config.index_topk;
        let hd_f: f32 = head_dim as f32;
        let sqrt_hd = hd_f.sqrt();
        let softmax_scale = 1.0_f32 / sqrt_hd;

        let compressor = Compressor::new(
            device.clone(),
            config,
            layer_id,
            compress_ratio,
            head_dim,
            true,
            max_batch,
            loader,
            cublas.clone(),
            Arc::clone(&kernels),
        )?;

        let n_comp = max_seqlen / compress_ratio;
        let kv_cache = GpuTensor::zeros(
            device.clone(),
            vec![max_batch, n_comp, head_dim],
            DType::BF16,
        )?;
        let kv_cache_cpu = vec![half::bf16::from_f32(0.0); max_batch * n_comp * head_dim];

        Ok(Self {
            wq_b,
            weights_proj,
            compressor,
            kv_cache,
            n_heads,
            head_dim,
            rope_head_dim,
            index_topk,
            compress_ratio,
            softmax_scale,
            device,
            cublas,
            kv_cache_cpu,
            kernels,
        })
    }

    fn gpu_bf16_to_f32(&self, gpu: &GpuTensor, total: usize, dim: usize) -> Result<Vec<f32>> {
        if gpu.dtype == DType::BF16 {
            let kernel_name = match dim {
                4096 => Some("cast_bf16_to_f32_N4096"),
                16384 => Some("cast_bf16_to_f32_N16384"),
                _ => None,
            };
            if let Some(kname) = kernel_name {
                let x_2d = GpuTensor {
                    slice: gpu.slice.clone(),
                    shape: vec![total, dim],
                    dtype: DType::BF16,
                    device: gpu.device.clone(),
                };
                let x_f32 = GpuTensor::zeros(self.device.clone(), vec![total, dim], DType::FP32)?;
                if self.kernels.call(kname, &[&x_2d, &x_f32]).is_ok() {
                    let host = x_f32.to_host()?;
                    return Ok(bytemuck::cast_slice(&host.data).to_vec());
                }
            }
        }

        let host = gpu.to_host()?;
        let bf16: &[half::bf16] = bytemuck::cast_slice(&host.data);
        Ok(bf16.iter().map(|v| v.to_f32()).collect())
    }

    fn try_gpu_index_score(
        &self,
        q_proj: &[f32],
        weights: &[f32],
        n_comp_tokens: usize,
        bsz: usize,
        seqlen: usize,
        start_pos: usize,
        offset: usize,
    ) -> Option<GpuTensor> {
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let _wq_out_dim = n_heads * head_dim;
        let total = bsz * seqlen;
        let topk = self.index_topk.min(n_comp_tokens);

        let kernel_name = format!(
            "indexer_score_h{}_d{}_topk{}",
            n_heads, head_dim, self.index_topk
        );

        let q_f32_cpu = CpuTensor::new(
            bytemuck::cast_slice(q_proj).to_vec(),
            vec![total, n_heads, head_dim],
            DType::FP32,
        );
        let q_gpu = GpuTensor::from_host(self.device.clone(), &q_f32_cpu).ok()?;

        let kv_f32: Vec<f32> = {
            let kv_bf16 = &self.kv_cache_cpu;
            let mut f32_data = vec![0.0f32; bsz * n_comp_tokens * head_dim];
            for (i, v) in kv_bf16.iter().enumerate() {
                if i < f32_data.len() {
                    f32_data[i] = v.to_f32();
                }
            }
            f32_data
        };
        let kv_cpu = CpuTensor::new(
            bytemuck::cast_slice(&kv_f32).to_vec(),
            vec![bsz, n_comp_tokens, head_dim],
            DType::FP32,
        );
        let kv_gpu = GpuTensor::from_host(self.device.clone(), &kv_cpu).ok()?;

        let w_cpu = CpuTensor::new(
            bytemuck::cast_slice(weights).to_vec(),
            vec![total, n_heads],
            DType::FP32,
        );
        let w_gpu = GpuTensor::from_host(self.device.clone(), &w_cpu).ok()?;

        let topk_out = GpuTensor::zeros(
            self.device.clone(),
            vec![total, topk],
            DType::INT32,
        ).ok()?;

        self.kernels.call(
            &kernel_name,
            &[&q_gpu, &kv_gpu, &w_gpu, &topk_out],
        ).ok()?;

        let mut causal_limit = vec![i32::MAX; total];
        if start_pos == 0 {
            for t in 0..total {
                let s = t % seqlen;
                causal_limit[t] = ((s + 1) / self.compress_ratio) as i32;
            }
        }
        let cl_cpu = CpuTensor::new(
            bytemuck::cast_slice(&causal_limit).to_vec(),
            vec![total],
            DType::INT32,
        );
        let cl_gpu = GpuTensor::from_host(self.device.clone(), &cl_cpu).ok()?;

        let off_cpu = CpuTensor::new(
            bytemuck::cast_slice(&[offset as i32]).to_vec(),
            vec![1],
            DType::INT32,
        );
        let off_gpu = GpuTensor::from_host(self.device.clone(), &off_cpu).ok()?;

        let adjusted = GpuTensor::zeros(
            self.device.clone(),
            vec![total, topk],
            DType::INT32,
        ).ok()?;

        self.kernels.call(
            "indexer_causal_adjust_topk512",
            &[&topk_out, &cl_gpu, &off_gpu, &adjusted],
        ).ok()?;

        Some(GpuTensor {
            slice: adjusted.slice,
            shape: vec![bsz, seqlen, topk],
            dtype: DType::INT32,
            device: adjusted.device,
        })
    }

    pub fn forward(
        &mut self,
        x: &GpuTensor,
        qr: &GpuTensor,
        start_pos: usize,
        offset: usize,
        bsz: usize,
        seqlen: usize,
    ) -> Result<GpuTensor> {
        let ratio = self.compress_ratio;
        let rd = self.rope_head_dim;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let end_pos = start_pos + seqlen;
        let wq_out_dim = n_heads * head_dim;

        let total = bsz * seqlen;
        let q_lora_rank = qr.shape[qr.shape.len() - 1];

        let qr_2d = GpuTensor {
            slice: qr.slice.clone(),
            shape: vec![total, q_lora_rank],
            dtype: qr.dtype,
            device: qr.device.clone(),
        };
        let mut q_gpu = GpuTensor::zeros(self.device.clone(), vec![total, wq_out_dim], DType::BF16)?;
        self.cublas.gemm_bf16(total, wq_out_dim, q_lora_rank, &qr_2d, &self.wq_b, &mut q_gpu, 1.0, 0.0)?;

        let mut q_proj = self.gpu_bf16_to_f32(&q_gpu, total, wq_out_dim)?;

        let half_rd = rd / 2;
        let rope = &self.compressor.rope;
        let (cos_data, sin_data) = rope.get_slice(start_pos, seqlen);
        for t in 0..total {
            let s = t % seqlen;
            for h in 0..n_heads {
                let base = t * wq_out_dim + h * head_dim;
                let rope_start = head_dim - rd;
                for k in 0..half_rd {
                    let idx1 = base + rope_start + 2 * k;
                    let idx2 = base + rope_start + 2 * k + 1;
                    let c = cos_data[s * half_rd + k] as f64;
                    let sn = sin_data[s * half_rd + k] as f64;
                    let v1 = q_proj[idx1] as f64;
                    let v2 = q_proj[idx2] as f64;
                    q_proj[idx1] = (v1 * c - v2 * sn) as f32;
                    q_proj[idx2] = (v1 * sn + v2 * c) as f32;
                }
            }
        }

        for t in 0..total {
            for h in 0..n_heads {
                let base = t * wq_out_dim + h * head_dim;
                crate::quant::hadamard_transform(&mut q_proj[base..base + head_dim], head_dim);
                crate::quant::fp4_act_quant_qdq(&mut q_proj[base..base + head_dim], head_dim, 32);
            }
        }

        let _compressed = self.compressor.forward(x, start_pos, bsz, seqlen)?;

        if let Some(ref compressed) = _compressed {
            let comp_host = compressed.to_host()?;
            let comp_bf16: &[half::bf16] = bytemuck::cast_slice(&comp_host.data);

            let n_comp_tokens_out = comp_host.shape.get(1).copied().unwrap_or(1);
            let kv_total_cols = self.kv_cache.shape[1];

            let write_start = if start_pos == 0 { 0 } else { start_pos / ratio };
            for b in 0..bsz {
                let dst_base = b * kv_total_cols * head_dim + write_start * head_dim;
                for g in 0..n_comp_tokens_out {
                    let src_off = (b * n_comp_tokens_out + g) * head_dim;
                    let dst_off = dst_base + g * head_dim;
                    if dst_off + head_dim <= self.kv_cache_cpu.len() && src_off + head_dim <= comp_bf16.len() {
                        for dd in 0..head_dim {
                            self.kv_cache_cpu[dst_off + dd] = comp_bf16[src_off + dd];
                        }
                    }
                }
            }

            let elem_size = 2usize;
            let src_batch_stride = n_comp_tokens_out * head_dim * elem_size;
            let dst_batch_stride = kv_total_cols * head_dim * elem_size;
            let copy_bytes = n_comp_tokens_out * head_dim * elem_size;
            let dst_offset = write_start * head_dim * elem_size;
            if let Err(e) = GpuTensor::d2d_scatter_rows(
                compressed,
                &mut self.kv_cache,
                src_batch_stride,
                dst_batch_stride,
                copy_bytes,
                dst_offset,
                bsz,
            ) {
                let out_cpu = CpuTensor::new(
                    bytemuck::cast_slice(&self.kv_cache_cpu).to_vec(),
                    self.kv_cache.shape.clone(),
                    DType::BF16,
                );
                self.kv_cache = GpuTensor::from_host(self.device.clone(), &out_cpu)?;
                eprintln!("warning: D2D scatter failed, fell back to H2D: {}", e);
            }
        }

        let dim = x.shape[x.shape.len() - 1];
        let x_2d = GpuTensor {
            slice: x.slice.clone(),
            shape: vec![total, dim],
            dtype: x.dtype,
            device: x.device.clone(),
        };
        let mut weights_gpu = GpuTensor::zeros(self.device.clone(), vec![total, n_heads], DType::BF16)?;
        self.cublas.gemm_bf16(total, n_heads, dim, &x_2d, &self.weights_proj, &mut weights_gpu, 1.0, 0.0)?;

        let weights_f32 = self.gpu_bf16_to_f32(&weights_gpu, total, n_heads)?;
        let scale_factor = self.softmax_scale * (n_heads as f32).powf(-0.5);
        let weights: Vec<f32> = weights_f32.iter().map(|v| v * scale_factor).collect();

        let kv_bf16 = &self.kv_cache_cpu;
        let n_comp_tokens = end_pos / ratio;
        let kv_total_cols = self.kv_cache.shape[1];

        if let Some(gpu_result) = self.try_gpu_index_score(
            &q_proj, &weights, n_comp_tokens, bsz, seqlen, start_pos, offset,
        ) {
            return Ok(gpu_result);
        }

        let mut index_score = vec![0.0f32; bsz * seqlen * n_comp_tokens];
        for b in 0..bsz {
            for s in 0..seqlen {
                for t in 0..n_comp_tokens {
                    let mut dot_sum = 0.0f32;
                    for h in 0..n_heads {
                        let q_base = (b * seqlen + s) * wq_out_dim + h * head_dim;
                        let kv_base = b * kv_total_cols * head_dim + t * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim {
                            let q_val = q_proj[q_base + dd];
                            let kv_val = if kv_base + dd < kv_bf16.len() {
                                kv_bf16[kv_base + dd].to_f32()
                            } else {
                                0.0
                            };
                            dot += q_val * kv_val;
                        }
                        let w = weights[(b * seqlen + s) * n_heads + h];
                        dot_sum += dot.max(0.0) * w;
                    }
                    index_score[(b * seqlen + s) * n_comp_tokens + t] = dot_sum;
                }
            }
        }

        if start_pos == 0 {
            for b in 0..bsz {
                for s in 0..seqlen {
                    for t in 0..n_comp_tokens {
                        if t >= (s + 1) / ratio {
                            index_score[(b * seqlen + s) * n_comp_tokens + t] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }

        let topk = self.index_topk.min(n_comp_tokens);
        let mut topk_idxs = vec![0i32; bsz * seqlen * topk];
        for b in 0..bsz {
            for s in 0..seqlen {
                let score_base = (b * seqlen + s) * n_comp_tokens;
                let mut idx: Vec<usize> = (0..n_comp_tokens).collect();
                idx.sort_by(|&a, &b| {
                    index_score[score_base + b].partial_cmp(&index_score[score_base + a]).unwrap()
                });
                for k in 0..topk {
                    let raw_idx = idx[k] as i32;
                    if start_pos == 0 && raw_idx >= ((s + 1) / ratio) as i32 {
                        topk_idxs[(b * seqlen + s) * topk + k] = -1;
                    } else {
                        topk_idxs[(b * seqlen + s) * topk + k] = raw_idx + offset as i32;
                    }
                }
            }
        }

        let cpu = CpuTensor::new(
            bytemuck::cast_slice(&topk_idxs).to_vec(),
            vec![bsz, seqlen, topk],
            DType::INT32,
        );
        GpuTensor::from_host(self.device.clone(), &cpu)
    }
}
