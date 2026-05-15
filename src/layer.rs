use crate::compressor::Compressor;
use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::expert::ExpertScheduler;
use crate::gate::Gate;
use crate::indexer::Indexer;
use crate::kv_cache::KvCache;
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaContext, DevicePtr};
use std::sync::Arc;

pub struct LayerWeights {
    pub wq_a: GpuTensor,
    pub wq_a_s: GpuTensor,
    pub wq_b: GpuTensor,
    pub wq_b_s: GpuTensor,
    pub wkv: GpuTensor,
    pub wkv_s: GpuTensor,
    pub wo_a: GpuTensor,
    pub wo_a_s: GpuTensor,
    pub wo_a_dequant: CpuTensor,
    pub wo_a_gpu: Option<GpuTensor>,
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
            Some(Compressor::new(
                device.clone(),
                &config,
                layer_id,
                compress_ratio,
                config.head_dim,
                false,
                max_batch,
                loader,
                cublas.clone(),
                Arc::clone(&kernels),
            )?)
        } else {
            None
        };

        let indexer = if compress_ratio > 0 && compress_ratio <= 4 {
            Some(Indexer::new(
                device.clone(),
                &config,
                layer_id,
                compress_ratio,
                max_batch,
                config.max_position_embeddings,
                loader,
                cublas.clone(),
                Arc::clone(&kernels),
            )?)
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

        let wo_a_gpu = if wo_a_is_bf16 {
            Some(GpuTensor::from_host(device.clone(), &wo_a_dequant)?)
        } else {
            None
        };

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

        let hc_attn_fn = load_gpu(loader, &(p.clone() + "hc_attn_fn"))?;
        let hc_attn_scale = load_gpu(loader, &(p.clone() + "hc_attn_scale"))?;
        let hc_attn_base = load_gpu(loader, &(p.clone() + "hc_attn_base"))?;
        let hc_ffn_fn = load_gpu(loader, &(p.clone() + "hc_ffn_fn"))?;
        let hc_ffn_scale = load_gpu(loader, &(p.clone() + "hc_ffn_scale"))?;
        let hc_ffn_base = load_gpu(loader, &(p.clone() + "hc_ffn_base"))?;

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
            wq_a,
            wq_a_s,
            wq_b,
            wq_b_s,
            wkv,
            wkv_s,
            wo_a: wo_a_raw,
            wo_a_s,
            wo_a_dequant,
            wo_a_gpu,
            wo_b,
            wo_b_s,
            q_norm,
            kv_norm,
            attn_norm,
            ffn_norm,
            attn_sink,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            shared_w1,
            shared_w1_s,
            shared_w3,
            shared_w3_s,
            shared_w2,
            shared_w2_s,
            gate_weight,
            gate_bias,
            wo_a_is_bf16,
        })
    }

    pub fn forward(
        &mut self,
        x: &GpuTensor,
        start_pos: usize,
        rope: &RopeCache,
        input_ids: Option<&[u32]>,
    ) -> Result<GpuTensor> {
        let residual = x.clone();

        let (x_attn, post, comb) =
            self.hc_pre(x, &self.weights.hc_attn_fn, &self.weights.hc_attn_scale, &self.weights.hc_attn_base)?;

        let x_norm = self.rmsnorm(&x_attn, Some(&self.weights.attn_norm))?;

        let attn_out = self.attention(&x_norm, start_pos, rope)?;

        let x = self.hc_post(&attn_out, &residual, &post, &comb)?;

        let residual = x.clone();

        let (x_ffn, post, comb) =
            self.hc_pre(&x, &self.weights.hc_ffn_fn, &self.weights.hc_ffn_scale, &self.weights.hc_ffn_base)?;

        let x_norm = self.rmsnorm(&x_ffn, Some(&self.weights.ffn_norm))?;

        let ffn_out = self.ffn(&x_norm, input_ids)?;

        let x = self.hc_post(&ffn_out, &residual, &post, &comb)?;

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
        let dim = self.config.hidden_size;
        let eps = self.config.hc_eps as f32;
        let sinkhorn_iters = self.config.hc_sinkhorn_iters;
        let device = x.device.clone();

        let x_flat_shape = vec![bsz * seqlen, hc * dim];
        let x_flat = self.reshape(x, &x_flat_shape)?;

        let x_flat_f32 = self.cast_to_f32(&x_flat)?;
        let x_normed = self.rmsnorm_f32(&x_flat_f32)?;

        let hc_fn_f32 = self.cast_to_f32(hc_fn)?;
        let mixes = self.gemm_f32(&x_normed, &hc_fn_f32)?;

        let total = bsz * seqlen;
        let mix_hc = (2 + hc) * hc;

        let pre_out = GpuTensor::zeros(device.clone(), vec![total, hc], DType::FP32)?;
        let post_out = GpuTensor::zeros(device.clone(), vec![total, hc], DType::FP32)?;
        let comb_out = GpuTensor::zeros(device.clone(), vec![total, hc, hc], DType::FP32)?;

        let mixes_2d = GpuTensor {
            slice: mixes.slice.clone(),
            shape: vec![total, mix_hc],
            dtype: mixes.dtype,
            device: device.clone(),
        };

        if self.kernels.call(
            "hc_sinkhorn_hc4_it20",
            &[&mixes_2d, &hc_scale, &hc_base, &pre_out, &post_out, &comb_out],
        ).is_ok() {
            let pre = GpuTensor { slice: pre_out.slice, shape: vec![bsz, seqlen, hc], dtype: DType::FP32, device: device.clone() };
            let post = GpuTensor { slice: post_out.slice, shape: vec![bsz, seqlen, hc], dtype: DType::FP32, device: device.clone() };
            let comb = GpuTensor { slice: comb_out.slice, shape: vec![bsz, seqlen, hc, hc], dtype: DType::FP32, device: device.clone() };

            let y = self.hc_reduce(x, &pre)?;
            return Ok((y, post, comb));
        }

        let mixes_host = mixes.to_host()?;
        let mixes_f32: &[f32] = bytemuck::cast_slice(&mixes_host.data);

        let scale_host = hc_scale.to_host()?;
        let scale_f32: &[f32] = bytemuck::cast_slice(&scale_host.data);

        let base_host = hc_base.to_host()?;
        let base_f32: &[f32] = bytemuck::cast_slice(&base_host.data);

        let mut pre_data = vec![0.0f32; total * hc];
        let mut post_data = vec![0.0f32; total * hc];
        let mut comb_data = vec![0.0f32; total * hc * hc];

        for t in 0..total {
            let m_base = t * mix_hc;
            for j in 0..hc {
                let mix_val = mixes_f32[m_base + j];
                let s = if 0 < scale_f32.len() { scale_f32[0] } else { 1.0 };
                let b = if j < base_f32.len() { base_f32[j] } else { 0.0 };
                pre_data[t * hc + j] = 1.0 / (1.0 + (-(mix_val * s + b)).exp()) + eps;
            }
            for j in 0..hc {
                let mix_val = mixes_f32[m_base + hc + j];
                let s = if scale_f32.len() > 1 { scale_f32[1] } else if !scale_f32.is_empty() { scale_f32[0] } else { 1.0 };
                let b = if hc + j < base_f32.len() { base_f32[hc + j] } else { 0.0 };
                post_data[t * hc + j] = 2.0 / (1.0 + (-(mix_val * s + b)).exp());
            }
            for j in 0..hc {
                for k in 0..hc {
                    let mix_val = mixes_f32[m_base + 2 * hc + j * hc + k];
                    let s = if scale_f32.len() > 2 { scale_f32[2] } else if !scale_f32.is_empty() { scale_f32[0] } else { 1.0 };
                    let b = if 2 * hc + j * hc + k < base_f32.len() { base_f32[2 * hc + j * hc + k] } else { 0.0 };
                    comb_data[t * hc * hc + j * hc + k] = mix_val * s + b;
                }
            }

            let sinkhorn_eps: f32 = 1e-6;
            {
                let mut row_max = vec![f32::NEG_INFINITY; hc];
                for j in 0..hc {
                    for k in 0..hc {
                        let v = comb_data[t * hc * hc + j * hc + k];
                        if v > row_max[j] { row_max[j] = v; }
                    }
                }
                let mut row_sum = vec![0.0f32; hc];
                for j in 0..hc {
                    for k in 0..hc {
                        let v = (comb_data[t * hc * hc + j * hc + k] - row_max[j]).exp();
                        comb_data[t * hc * hc + j * hc + k] = v;
                        row_sum[j] += v;
                    }
                }
                for j in 0..hc {
                    for k in 0..hc {
                        comb_data[t * hc * hc + j * hc + k] =
                            comb_data[t * hc * hc + j * hc + k] / row_sum[j] + sinkhorn_eps;
                    }
                }
                let mut col_sum = vec![0.0f32; hc];
                for j in 0..hc {
                    for k in 0..hc {
                        col_sum[k] += comb_data[t * hc * hc + j * hc + k];
                    }
                }
                for j in 0..hc {
                    for k in 0..hc {
                        comb_data[t * hc * hc + j * hc + k] /= col_sum[k] + sinkhorn_eps;
                    }
                }
            }
            for _iter in 1..sinkhorn_iters {
                let mut row_sum = vec![0.0f32; hc];
                for j in 0..hc {
                    for k in 0..hc {
                        row_sum[j] += comb_data[t * hc * hc + j * hc + k];
                    }
                }
                for j in 0..hc {
                    for k in 0..hc {
                        comb_data[t * hc * hc + j * hc + k] /= row_sum[j] + sinkhorn_eps;
                    }
                }
                let mut col_sum = vec![0.0f32; hc];
                for j in 0..hc {
                    for k in 0..hc {
                        col_sum[k] += comb_data[t * hc * hc + j * hc + k];
                    }
                }
                for j in 0..hc {
                    for k in 0..hc {
                        comb_data[t * hc * hc + j * hc + k] /= col_sum[k] + sinkhorn_eps;
                    }
                }
            }
        }

        let pre_cpu = CpuTensor::new(bytemuck::cast_slice(&pre_data).to_vec(), vec![bsz, seqlen, hc], DType::FP32);
        let post_cpu = CpuTensor::new(bytemuck::cast_slice(&post_data).to_vec(), vec![bsz, seqlen, hc], DType::FP32);
        let comb_cpu = CpuTensor::new(bytemuck::cast_slice(&comb_data).to_vec(), vec![bsz, seqlen, hc, hc], DType::FP32);

        let pre_gpu = GpuTensor::from_host(device.clone(), &pre_cpu)?;
        let post_gpu = GpuTensor::from_host(device.clone(), &post_cpu)?;
        let comb_gpu = GpuTensor::from_host(device.clone(), &comb_cpu)?;

        let y = self.hc_reduce(x, &pre_gpu)?;

        Ok((y, post_gpu, comb_gpu))
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

        self.cublas.gemm_f32_nn_strided_batched(
            hc, dim, hc,
            &comb_3d, &residual_f32, &mut y_f32,
            (hc * hc) as i64, (hc * dim) as i64, (hc * dim) as i64,
            total as i32,
            1.0, 1.0,
        )?;

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
        let _rope_dim = self.config.qk_rope_head_dim;
        let kv_dim = self.config.kv_dim();
        let win = self.config.sliding_window;
        let ratio = self.config.compress_ratio(self.layer_id) as usize;
        let _device = x.device.clone();

        let qr = self.fp8_gemm_act_quant(x, &self.weights.wq_a, &self.weights.wq_a_s, self.config.q_lora_rank)?;
        let qr = self.reshape(&qr, &[bsz, seqlen, self.config.q_lora_rank])?;
        let q_normed = self.rmsnorm(&qr, Some(&self.weights.q_norm))?;
        let q_normed_flat = self.reshape(&q_normed, &[bsz * seqlen, self.config.q_lora_rank])?;
        let q = self.fp8_gemm_act_quant(&q_normed_flat, &self.weights.wq_b, &self.weights.wq_b_s, n_heads * head_dim)?;

        let q = self.reshape(&q, &[bsz, seqlen, n_heads, head_dim])?;
        let q = self.rmsnorm(&q, None)?;

        let q_rotated = self.apply_rope_q(&q, start_pos, rope)?;

        let kv = self.fp8_gemm_act_quant(x, &self.weights.wkv, &self.weights.wkv_s, kv_dim)?;
        let kv = self.reshape(&kv, &[bsz, seqlen, kv_dim])?;
        let kv_normed = self.rmsnorm(&kv, Some(&self.weights.kv_norm))?;
        let kv_rotated = self.apply_rope_kv(&kv_normed, start_pos, rope)?;
        let kv_final = self.act_quant_inplace_nope(&kv_rotated)?;

        let topk_idxs = self.compute_topk_idxs(bsz, seqlen, start_pos)?;

        if start_pos == 0 {
            self.kv_cache.update_prefill(&kv_final, 0, seqlen)?;

            if ratio > 0 {
                if let Some(ref mut compressor) = self.compressor {
                    if let Some(compressed) = compressor.forward(x, start_pos, bsz, seqlen)? {
                        self.kv_cache.write_compressed(&compressed, 0, start_pos, seqlen)?;
                    }
                }

                let offset = seqlen;
                let compress_topk = if let Some(ref mut indexer) = self.indexer {
                    indexer.forward(x, &q_normed, start_pos, offset, bsz, seqlen)?
                } else {
                    self.get_compress_topk_uniform(ratio, bsz, seqlen, start_pos, offset)?
                };
                let combined_topk = self.concat_topk(&topk_idxs, &compress_topk, bsz, seqlen)?;

                let compressed_kv = self.kv_cache.get_compressed_kv(0, seqlen)?;
                let full_kv = self.concat_kv(&kv_final, &compressed_kv, bsz, seqlen)?;

                let attn_out = self.sparse_attention(&q_rotated, &full_kv, &self.weights.attn_sink, &combined_topk, bsz, seqlen, start_pos)?;
                let attn_out = self.apply_inverse_rope(&attn_out, start_pos, rope)?;
                return self.output_proj(&attn_out);
            }

            let attn_out = self.sparse_attention(&q_rotated, &kv_final, &self.weights.attn_sink, &topk_idxs, bsz, seqlen, start_pos)?;
            let attn_out = self.apply_inverse_rope(&attn_out, start_pos, rope)?;
            return self.output_proj(&attn_out);
        }

        self.kv_cache.update_decode(&kv_final, 0, start_pos)?;

        if ratio > 0 {
            if let Some(ref mut compressor) = self.compressor {
                if let Some(compressed) = compressor.forward(x, start_pos, bsz, seqlen)? {
                    self.kv_cache.write_compressed(&compressed, 0, start_pos, seqlen)?;
                }
            }

            let offset = win;
            let compress_topk = if let Some(ref mut indexer) = self.indexer {
                indexer.forward(x, &q_normed, start_pos, offset, bsz, seqlen)?
            } else {
                self.get_compress_topk_uniform(ratio, bsz, seqlen, start_pos, offset)?
            };
            let combined_topk = self.concat_topk(&topk_idxs, &compress_topk, bsz, seqlen)?;

            let full_cache = self.kv_cache.get_full_cache(0)?;
            let attn_out = self.sparse_attention(&q_rotated, &full_cache, &self.weights.attn_sink, &combined_topk, bsz, seqlen, start_pos)?;
            let attn_out = self.apply_inverse_rope(&attn_out, start_pos, rope)?;
            return self.output_proj(&attn_out);
        }

        let full_cache = self.kv_cache.get_full_cache(0)?;
        let attn_out = self.sparse_attention(&q_rotated, &full_cache, &self.weights.attn_sink, &topk_idxs, bsz, seqlen, start_pos)?;
        let attn_out = self.apply_inverse_rope(&attn_out, start_pos, rope)?;
        self.output_proj(&attn_out)
    }

    fn output_proj(&self, o: &GpuTensor) -> Result<GpuTensor> {
        let bsz = o.shape[0];
        let seqlen = o.shape[1];
        let n_groups = self.config.o_groups;
        let o_lora_rank = self.config.o_lora_rank;
        let n_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let group_dim = n_heads * head_dim / n_groups;
        let dim = self.config.hidden_size;
        let device = o.device.clone();
        let total = bsz * seqlen;

        if self.weights.wo_a_is_bf16 {
            let o_2d = GpuTensor {
                slice: o.slice.clone(),
                shape: vec![total, n_groups, group_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };

            let wo_a_gpu = self.weights.wo_a_gpu.as_ref()
                .ok_or_else(|| anyhow::anyhow!("wo_a_gpu cache not initialized for bf16 path"))?;

            let mut result = GpuTensor::zeros(device.clone(), vec![total, n_groups, o_lora_rank], DType::BF16)?;

            self.cublas.gemm_bf16_nn_strided_batched(
                o_lora_rank, group_dim, group_dim,
                wo_a_gpu, &o_2d, &mut result,
                (o_lora_rank * group_dim) as i64,
                group_dim as i64,
                o_lora_rank as i64,
                (total * n_groups) as i32,
                1.0, 0.0,
            )?;

            let x = self.fp8_gemm_act_quant(&result, &self.weights.wo_b, &self.weights.wo_b_s, dim)?;
            let x = self.reshape(&x, &[bsz, seqlen, dim])?;
            return Ok(x);
        }

        let o_reshaped = self.reshape(o, &[bsz, seqlen, n_groups, group_dim])?;
        let o_host = o_reshaped.to_host()?;
        let o_bf16: &[half::bf16] = bytemuck::cast_slice(&o_host.data);

        let wo_a_bf16: &[half::bf16] = bytemuck::cast_slice(&self.weights.wo_a_dequant.data);

        let mut result_data = vec![0u16; bsz * seqlen * n_groups * o_lora_rank];

        for g in 0..n_groups {
            let mut x_group = vec![0u16; bsz * seqlen * group_dim];
            for b in 0..bsz {
                for s in 0..seqlen {
                    let src_off = ((b * seqlen + s) * n_groups + g) * group_dim;
                    let dst_off = (b * seqlen + s) * group_dim;
                    for d in 0..group_dim {
                        if src_off + d < o_bf16.len() && dst_off + d < x_group.len() {
                            x_group[dst_off + d] = o_bf16[src_off + d].to_bits();
                        }
                    }
                }
            }

            let x_cpu = CpuTensor::new(bytemuck::cast_slice(&x_group).to_vec(), vec![bsz * seqlen, group_dim], DType::BF16);
            let x_gpu = GpuTensor::from_host(device.clone(), &x_cpu)?;

            let w_shape = vec![o_lora_rank, group_dim];
            let w_cpu = CpuTensor::new(
                bytemuck::cast_slice(&wo_a_bf16[g * o_lora_rank * group_dim..(g + 1) * o_lora_rank * group_dim]).to_vec(),
                w_shape,
                DType::BF16,
            );
            let w_gpu = GpuTensor::from_host(device.clone(), &w_cpu)?;

            let mut c_gpu = GpuTensor::zeros(device.clone(), vec![bsz * seqlen, o_lora_rank], DType::BF16)?;
            self.cublas.gemm_bf16(bsz * seqlen, o_lora_rank, group_dim, &x_gpu, &w_gpu, &mut c_gpu, 1.0, 0.0)?;

            let c_host = c_gpu.to_host()?;
            let c_bf16: &[half::bf16] = bytemuck::cast_slice(&c_host.data);
            for b in 0..bsz {
                for s in 0..seqlen {
                    let src_off = (b * seqlen + s) * o_lora_rank;
                    let dst_off = ((b * seqlen + s) * n_groups + g) * o_lora_rank;
                    for d in 0..o_lora_rank {
                        if src_off + d < c_bf16.len() && dst_off + d < result_data.len() {
                            result_data[dst_off + d] = c_bf16[src_off + d].to_bits();
                        }
                    }
                }
            }
        }

        let result_cpu = CpuTensor::new(
            bytemuck::cast_slice(&result_data).to_vec(),
            vec![bsz, seqlen, n_groups * o_lora_rank],
            DType::BF16,
        );
        let result_gpu = GpuTensor::from_host(device.clone(), &result_cpu)?;

        let x = self.fp8_gemm_act_quant(&result_gpu, &self.weights.wo_b, &self.weights.wo_b_s, dim)?;
        let x = self.reshape(&x, &[bsz, seqlen, dim])?;

        Ok(x)
    }

    fn ffn_shared(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = self.config.hidden_size;
        let inter_dim = self.config.moe_intermediate_size;
        let _device = x.device.clone();

        let x_flat = self.reshape(x, &[bsz * seqlen, dim])?;

        let gate = self.fp8_gemm_act_quant(&x_flat, &self.weights.shared_w1, &self.weights.shared_w1_s, inter_dim)?;
        let up = self.fp8_gemm_act_quant(&x_flat, &self.weights.shared_w3, &self.weights.shared_w3_s, inter_dim)?;

        let swiglu_out = self.swiglu(&gate, &up)?;

        let down = self.fp8_gemm_act_quant(&swiglu_out, &self.weights.shared_w2, &self.weights.shared_w2_s, dim)?;
        let out = self.reshape(&down, &[bsz, seqlen, dim])?;

        Ok(out)
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

        let mut y_gpu = shared_out;

        let x_flat = self.reshape(x, &[total, dim])?;

        let indices_host = gate_output.indices.to_host()?;
        let indices_i32: &[i32] = bytemuck::cast_slice(&indices_host.data);

        let mut expert_counts = vec![0usize; n_experts];
        for &idx in indices_i32 {
            let e = idx as usize;
            if e < n_experts {
                expert_counts[e] += 1;
            }
        }

        let use_scatter_add = dim == 4096 || dim == 7168;

        for expert_id in 0..n_experts {
            if expert_counts[expert_id] == 0 {
                continue;
            }

            let mut expert_tokens = Vec::new();
            for t in 0..total {
                for k in 0..topk {
                    if indices_i32[t * topk + k] as usize == expert_id {
                        expert_tokens.push((t, k));
                    }
                }
            }

            if expert_tokens.is_empty() {
                continue;
            }

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
                if self.kernels.call(&kernel_name, &[&x_src, &tid_gpu, &x_dst]).is_ok() {
                    x_dst
                } else {
                    x_flat.gather_rows(&row_indices, dim)?
                }
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
                if !kernel_name.is_empty() && self.kernels.call(kernel_name, &[&expert_2d, &w_gpu, &tid_gpu, &y_2d]).is_ok() {
                    y_gpu = GpuTensor {
                        slice: y_2d.slice,
                        shape: vec![bsz, seqlen, dim],
                        dtype: DType::BF16,
                        device: device.clone(),
                    };
                    continue;
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
            self.expert_scheduler.prefetch_next_layer(self.layer_id, &active_expert_ids, &mut self.weight_loader)?;
        }

        Ok(y_gpu)
    }

    fn compute_expert(&mut self, x: &GpuTensor, expert_id: usize) -> Result<GpuTensor> {
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

        if self.expert_scheduler.expert_dtype == DType::FP4E2M1 {
            let expert = self.expert_scheduler.get_expert_gpu_raw(self.layer_id, expert_id)?;
            let gate = self.fp4_gemm_act_quant(&x_flat, &expert.w1, &expert.w1_scale, inter_dim)?;
            let up = self.fp4_gemm_act_quant(&x_flat, &expert.w3, &expert.w3_scale, inter_dim)?;
            let swiglu_out = self.swiglu(&gate, &up)?;
            let swiglu_flat = GpuTensor {
                slice: swiglu_out.slice.clone(),
                shape: vec![m, inter_dim],
                dtype: swiglu_out.dtype,
                device: device.clone(),
            };
            let down = self.fp4_gemm_act_quant(&swiglu_flat, &expert.w2, &expert.w2_scale, dim)?;
            Ok(down)
        } else {
            let expert = self.expert_scheduler.get_expert_gpu(self.layer_id, expert_id)?;
            let gate = self.fp8_gemm_act_quant(&x_flat, &expert.w1, &expert.w1_scale, inter_dim)?;
            let up = self.fp8_gemm_act_quant(&x_flat, &expert.w3, &expert.w3_scale, inter_dim)?;
            let swiglu_out = self.swiglu(&gate, &up)?;
            let swiglu_flat = GpuTensor {
                slice: swiglu_out.slice.clone(),
                shape: vec![m, inter_dim],
                dtype: swiglu_out.dtype,
                device: device.clone(),
            };
            let down = self.fp8_gemm_act_quant(&swiglu_flat, &expert.w2, &expert.w2_scale, dim)?;
            Ok(down)
        }
    }

    fn act_quant_inplace_nope(&self, kv: &GpuTensor) -> Result<GpuTensor> {
        let device = kv.device.clone();
        let shape = kv.shape.clone();
        let head_dim = self.config.head_dim;
        let rope_dim = self.config.qk_rope_head_dim;
        let nope_dim = head_dim - rope_dim;
        let block_size = 64usize;
        let fp8_max = 448.0f32;

        if nope_dim == 448 && block_size == 64 {
            let kernel_name = "act_quant_N448_bs64_inplace";
            let _last_dim = shape.last().copied().unwrap_or(1);
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

            if self.kernels.call(kernel_name, &[&nope_2d, &y_nope, &s]).is_ok() {
                let rope_gpu = self.slice_columns(kv, nope_dim, head_dim)?;
                return self.concat_columns(&y_nope, &rope_gpu, &shape);
            }
        }

        let kv_host = kv.to_host()?;
        let kv_bf16: &[half::bf16] = bytemuck::cast_slice(&kv_host.data);
        let mut out_data = kv_bf16.to_vec();
        let last_dim = shape.last().copied().unwrap_or(1);
        let n_rows = kv_bf16.len() / last_dim;

        for r in 0..n_rows {
            let base = r * last_dim;
            for block_start in (0..nope_dim).step_by(block_size) {
                let block_end = (block_start + block_size).min(nope_dim);
                let mut amax: f32 = 0.0;
                for d in block_start..block_end {
                    let v = out_data[base + d].to_f32();
                    if v.abs() > amax { amax = v.abs(); }
                }
                amax = amax.max(1e-4);
                let scale = amax / fp8_max;
                for d in block_start..block_end {
                    let v = out_data[base + d].to_f32();
                    let q = v / scale;
                    let fp8_bits = crate::quant::f32_to_fp8_e4m3(q);
                    let deq = crate::quant::fp8_e4m3_to_f32(fp8_bits);
                    out_data[base + d] = half::bf16::from_f32(deq * scale);
                }
            }
        }

        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out_data).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn rmsnorm(&self, x: &GpuTensor, weight: Option<&GpuTensor>) -> Result<GpuTensor> {
        let device = x.device.clone();
        let shape = x.shape.clone();
        let eps = self.config.rms_norm_eps as f32;
        let last_dim = shape.last().copied().unwrap_or(1);

        let kernel_name = if weight.is_some() {
            match last_dim {
                4096 => Some("rmsnorm_N4096"),
                1024 => Some("rmsnorm_N1024"),
                512 => Some("rmsnorm_N512"),
                _ => None,
            }
        } else {
            if last_dim == 1024 { Some("rmsnorm_no_weight_N1024") } else { None }
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

            if self.kernels.call(name, &[&x_2d, &w_f32, &y]).is_ok() {
                return Ok(GpuTensor {
                    slice: y.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let x_host = x.to_host()?;
        let x_bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);

        let w_host = weight.map(|w| w.to_host()).transpose()?;
        let w_bf16: Option<&[half::bf16]> = w_host.as_ref().map(|h| bytemuck::cast_slice(&h.data));

        let out = crate::quant::rmsnorm_bf16(x_bf16, w_bf16, last_dim, eps);
        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn rmsnorm_f32(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let device = x.device.clone();
        let shape = x.shape.clone();
        let eps = self.config.rms_norm_eps as f32;
        let last_dim = shape.last().copied().unwrap_or(1);
        let n = shape.iter().product::<usize>();
        let m = n / last_dim;

        let kernel_name = match last_dim {
            4096 => Some("rmsnorm_f32_N4096"),
            7168 => Some("rmsnorm_f32_N7168"),
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
            if self.kernels.call(kname, &[&x_2d, &y_2d]).is_ok() {
                return Ok(GpuTensor {
                    slice: y_2d.slice,
                    shape,
                    dtype: DType::FP32,
                    device,
                });
            }
        }

        let x_host = x.to_host()?;
        let x_f32: &[f32] = bytemuck::cast_slice(&x_host.data);

        let out = crate::quant::rmsnorm_f32(x_f32, last_dim, eps);
        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out).to_vec(), shape, DType::FP32);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn fp8_gemm_act_quant(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        weight_scale: &GpuTensor,
        out_dim: usize,
    ) -> Result<GpuTensor> {
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        let n = out_dim;
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

        if kernel_name.is_some() && weight.dtype == DType::FP8E4M3 {
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

            if self.kernels.call(kernel_name.unwrap(), &[&x_2d, &w_2d, &c, &x_s_2d, &w_s_2d]).is_ok() {
                return Ok(c);
            }
        }

        let w_host = weight.to_host()?;
        let w_dequant = match w_host.dtype {
            DType::BF16 => CpuTensor::new(w_host.data.clone(), w_host.shape.clone(), DType::BF16),
            DType::FP8E4M3 => {
                let s_host = weight_scale.to_host()?;
                crate::quant::dequant_fp8_e4m3_to_bf16(&w_host.data, &s_host.data, &w_host.shape)?
            }
            _ => {
                return Err(anyhow!("unsupported weight dtype for gemm: {}", w_host.dtype));
            }
        };
        let w_gpu = GpuTensor::from_host(device.clone(), &w_dequant)?;

        let mut c = GpuTensor::zeros(device, vec![m, n], DType::BF16)?;
        self.cublas.gemm_bf16(m, n, k, x, &w_gpu, &mut c, 1.0, 0.0)?;

        Ok(c)
    }

    fn fp4_gemm_act_quant(
        &self,
        x: &GpuTensor,
        weight: &GpuTensor,
        weight_scale: &GpuTensor,
        out_dim: usize,
    ) -> Result<GpuTensor> {
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
        let n = out_dim;
        let device = x.device.clone();

        let kernel_name = match (n, k) {
            (2048, 4096) => Some("fp4_gemm_N2048_K4096"),
            (4096, 2048) => Some("fp4_gemm_N4096_K2048"),
            _ => None,
        };

        if kernel_name.is_some() && weight.dtype == DType::FP4E2M1 {
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

            if self.kernels.call(kernel_name.unwrap(), &[&x_2d, &w_2d, &c, &x_s_2d, &w_s_2d]).is_ok() {
                return Ok(c);
            }
        }

        let w_host = weight.to_host()?;
        let s_host = weight_scale.to_host()?;
        let w_dequant = match w_host.dtype {
            DType::BF16 => CpuTensor::new(w_host.data.clone(), w_host.shape.clone(), DType::BF16),
            DType::FP8E4M3 => crate::quant::dequant_fp8_e4m3_to_bf16(&w_host.data, &s_host.data, &w_host.shape)?,
            DType::FP4E2M1 => {
                let logical_k = n.max(k);
                crate::quant::dequant_fp4_e2m1_to_bf16(&w_host.data, &s_host.data, &w_host.shape, logical_k)?
            }
            _ => return Err(anyhow!("unsupported weight dtype for fp4_gemm: {}", w_host.dtype)),
        };
        let w_gpu = GpuTensor::from_host(device.clone(), &w_dequant)?;

        let mut c = GpuTensor::zeros(device, vec![m, n], DType::BF16)?;
        self.cublas.gemm_bf16(m, n, k, x, &w_gpu, &mut c, 1.0, 0.0)?;

        Ok(c)
    }

    fn act_quant_gpu(&self, x: &GpuTensor, block_size: usize) -> Result<(GpuTensor, GpuTensor)> {
        let device = x.device.clone();
        let m = x.shape.iter().rev().skip(1).product::<usize>();
        let k = *x.shape.last().unwrap_or(&1);
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
            let s = GpuTensor::zeros(device.clone(), vec![m, n_blocks], DType::FP32)?;

            if self.kernels.call(name, &[&x_2d, &y, &s]).is_ok() {
                return Ok((y, s));
            }
        }

        let x_host = x.to_host()?;
        let x_bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
        let fp8_max = 448.0f32;

        let mut y_data = vec![0u8; m * k];
        let mut s_data = vec![0.0f32; m * n_blocks];

        for r in 0..m {
            for b in 0..n_blocks {
                let block_start = b * block_size;
                let mut amax = 0.0f32;
                for d in 0..block_size {
                    let v = x_bf16[r * k + block_start + d].to_f32();
                    if v.abs() > amax { amax = v.abs(); }
                }
                amax = amax.max(1e-4);
                let scale = amax / fp8_max;
                s_data[r * n_blocks + b] = scale;
                for d in 0..block_size {
                    let v = x_bf16[r * k + block_start + d].to_f32();
                    let q = (v / scale).clamp(-fp8_max, fp8_max);
                    y_data[r * k + block_start + d] = crate::quant::f32_to_fp8_e4m3(q);
                }
            }
        }

        let y_cpu = CpuTensor::new(y_data, vec![m, k], DType::FP8E4M3);
        let y_gpu = GpuTensor::from_host(device.clone(), &y_cpu)?;

        let s_cpu = CpuTensor::new(bytemuck::cast_slice(&s_data).to_vec(), vec![m, n_blocks], DType::FP32);
        let s_gpu = GpuTensor::from_host(device, &s_cpu)?;

        Ok((y_gpu, s_gpu))
    }

    fn apply_rope_q(&self, q: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = q.device.clone();
        let shape = q.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let n_heads = shape[2];
        let head_dim = shape[3];
        let rope_dim = self.config.qk_rope_head_dim;
        let half_rope = rope_dim / 2;
        let nope_dim = head_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let kernel_name = "rope_interleaved_fwd_D64";

            let q_3d = GpuTensor {
                slice: q.slice.clone(),
                shape: vec![total, n_heads, head_dim],
                dtype: q.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&q_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&q_3d, nope_dim, head_dim)?;

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
            let cos_gpu = GpuTensor::from_host(device.clone(), &cos_cpu)?;
            let sin_gpu = GpuTensor::from_host(device.clone(), &sin_cpu)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, n_heads, rope_dim], DType::BF16)?;

            if self.kernels.call(kernel_name, &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope]).is_ok() {
                let out_shape = vec![total, n_heads, head_dim];
                let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
                return Ok(GpuTensor {
                    slice: result.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let q_host = q.to_host()?;
        let q_bf16: &[half::bf16] = bytemuck::cast_slice(&q_host.data);
        let (cos_data, sin_data) = rope.get_slice(start_pos, seqlen);

        let mut out_data = q_bf16.to_vec();
        for b in 0..bsz {
            for s in 0..seqlen {
                for h in 0..n_heads {
                    let base = (b * seqlen * n_heads + s * n_heads + h) * head_dim;
                    let rope_start = head_dim - rope_dim;
                    for k in 0..half_rope {
                        let idx1 = rope_start + 2 * k;
                        let idx2 = rope_start + 2 * k + 1;
                        let c = cos_data[s * half_rope + k] as f64;
                        let sn = sin_data[s * half_rope + k] as f64;
                        let v1 = out_data[base + idx1].to_f32() as f64;
                        let v2 = out_data[base + idx2].to_f32() as f64;
                        let r1 = v1 * c - v2 * sn;
                        let r2 = v1 * sn + v2 * c;
                        out_data[base + idx1] = half::bf16::from_f32(r1 as f32);
                        out_data[base + idx2] = half::bf16::from_f32(r2 as f32);
                    }
                }
            }
        }

        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out_data).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn apply_rope_kv(&self, kv: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = kv.device.clone();
        let shape = kv.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let kv_dim = shape[2];
        let rope_dim = self.config.qk_rope_head_dim;
        let half_rope = rope_dim / 2;
        let nope_dim = kv_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let kernel_name = "rope_interleaved_fwd_D64";
            let kv_3d = GpuTensor {
                slice: kv.slice.clone(),
                shape: vec![total, 1, kv_dim],
                dtype: kv.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&kv_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&kv_3d, nope_dim, kv_dim)?;

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
            let cos_gpu = GpuTensor::from_host(device.clone(), &cos_cpu)?;
            let sin_gpu = GpuTensor::from_host(device.clone(), &sin_cpu)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, 1, rope_dim], DType::BF16)?;

            if self.kernels.call(kernel_name, &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope]).is_ok() {
                let out_shape = vec![total, 1, kv_dim];
                let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
                return Ok(GpuTensor {
                    slice: result.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let kv_host = kv.to_host()?;
        let kv_bf16: &[half::bf16] = bytemuck::cast_slice(&kv_host.data);
        let (cos_data, sin_data) = rope.get_slice(start_pos, seqlen);

        let mut out_data = kv_bf16.to_vec();
        for b in 0..bsz {
            for s in 0..seqlen {
                let base = (b * seqlen + s) * kv_dim;
                let rope_start = kv_dim - rope_dim;
                for k in 0..half_rope {
                    let idx1 = rope_start + 2 * k;
                    let idx2 = rope_start + 2 * k + 1;
                    let c = cos_data[s * half_rope + k] as f64;
                    let sn = sin_data[s * half_rope + k] as f64;
                    let v1 = out_data[base + idx1].to_f32() as f64;
                    let v2 = out_data[base + idx2].to_f32() as f64;
                    let r1 = v1 * c - v2 * sn;
                    let r2 = v1 * sn + v2 * c;
                    out_data[base + idx1] = half::bf16::from_f32(r1 as f32);
                    out_data[base + idx2] = half::bf16::from_f32(r2 as f32);
                }
            }
        }

        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out_data).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn apply_inverse_rope(&self, o: &GpuTensor, start_pos: usize, rope: &RopeCache) -> Result<GpuTensor> {
        let device = o.device.clone();
        let shape = o.shape.clone();
        let bsz = shape[0];
        let seqlen = shape[1];
        let n_heads = shape[2];
        let head_dim = shape[3];
        let rope_dim = self.config.qk_rope_head_dim;
        let half_rope = rope_dim / 2;
        let nope_dim = head_dim - rope_dim;
        let total = bsz * seqlen;

        if rope_dim == 64 {
            let kernel_name = "rope_interleaved_inv_D64";
            let o_3d = GpuTensor {
                slice: o.slice.clone(),
                shape: vec![total, n_heads, head_dim],
                dtype: o.dtype,
                device: device.clone(),
            };

            let nope_gpu = self.slice_columns(&o_3d, 0, nope_dim)?;
            let rope_gpu = self.slice_columns(&o_3d, nope_dim, head_dim)?;

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
            let cos_gpu = GpuTensor::from_host(device.clone(), &cos_cpu)?;
            let sin_gpu = GpuTensor::from_host(device.clone(), &sin_cpu)?;

            let y_rope = GpuTensor::zeros(device.clone(), vec![total, n_heads, rope_dim], DType::BF16)?;

            if self.kernels.call(kernel_name, &[&rope_gpu, &cos_gpu, &sin_gpu, &y_rope]).is_ok() {
                let out_shape = vec![total, n_heads, head_dim];
                let result = self.concat_columns(&nope_gpu, &y_rope, &out_shape)?;
                return Ok(GpuTensor {
                    slice: result.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let o_host = o.to_host()?;
        let o_bf16: &[half::bf16] = bytemuck::cast_slice(&o_host.data);
        let (cos_data, sin_data) = rope.get_slice(start_pos, seqlen);

        let mut out_data = o_bf16.to_vec();
        for b in 0..bsz {
            for s in 0..seqlen {
                for h in 0..n_heads {
                    let base = (b * seqlen * n_heads + s * n_heads + h) * head_dim;
                    let rope_start = head_dim - rope_dim;
                    for k in 0..half_rope {
                        let idx1 = rope_start + 2 * k;
                        let idx2 = rope_start + 2 * k + 1;
                        let c = cos_data[s * half_rope + k] as f64;
                        let sn = sin_data[s * half_rope + k] as f64;
                        let v1 = out_data[base + idx1].to_f32() as f64;
                        let v2 = out_data[base + idx2].to_f32() as f64;
                        let r1 = v1 * c + v2 * sn;
                        let r2 = -v1 * sn + v2 * c;
                        out_data[base + idx1] = half::bf16::from_f32(r1 as f32);
                        out_data[base + idx2] = half::bf16::from_f32(r2 as f32);
                    }
                }
            }
        }

        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out_data).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
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
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let device = q.device.clone();

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

            if self.kernels.call(
                "sparse_attn_h64_d512",
                &[q, kv, &o, &sink_1d, topk_idxs],
            ).is_ok() {
                return Ok(o);
            }
        }

        let q_host = q.to_host()?;
        let kv_host = kv.to_host()?;
        let sink_host = attn_sink.to_host()?;
        let topk_host = topk_idxs.to_host()?;

        let q_bf16: &[half::bf16] = bytemuck::cast_slice(&q_host.data);
        let kv_bf16: &[half::bf16] = bytemuck::cast_slice(&kv_host.data);
        let sink_f32: Vec<f32> = if sink_host.dtype == DType::FP32 {
            bytemuck::cast_slice(&sink_host.data).to_vec()
        } else {
            let sink_bf16: &[half::bf16] = bytemuck::cast_slice(&sink_host.data);
            sink_bf16.iter().map(|v| v.to_f32()).collect()
        };
        let topk_i32: &[i32] = bytemuck::cast_slice(&topk_host.data);

        let kv_seqlen = kv.shape[1];
        let topk_count = topk_host.shape[2];
        let mut out_data = vec![0u16; bsz * seqlen * n_heads * head_dim];

        for b in 0..bsz {
            for s in 0..seqlen {
                for h in 0..n_heads {
                    let q_base = (b * seqlen * n_heads + s * n_heads + h) * head_dim;
                    let sink_val = if h < sink_f32.len() { sink_f32[h] } else { 0.0 };

                    let mut scores = Vec::with_capacity(topk_count);
                    let mut valid_positions = Vec::with_capacity(topk_count);
                    for k in 0..topk_count {
                        let idx = topk_i32[(b * seqlen + s) * topk_count + k];
                        if idx < 0 || idx as usize >= kv_seqlen {
                            continue;
                        }
                        let t = idx as usize;
                        let kv_base = (b * kv_seqlen + t) * head_dim;
                        let mut dot: f32 = 0.0;
                        for d in 0..head_dim {
                            dot += q_bf16[q_base + d].to_f32() * kv_bf16[kv_base + d].to_f32();
                        }
                        scores.push(dot * softmax_scale);
                        valid_positions.push(t);
                    }

                    if scores.is_empty() {
                        continue;
                    }

                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sink_exp: f32 = (sink_val - max_score).exp();
                    let exp_sum: f32 = scores.iter().map(|s: &f32| (s - max_score).exp()).sum::<f32>() + sink_exp;
                    let inv_sum = 1.0 / exp_sum;

                    let o_base = (b * seqlen * n_heads + s * n_heads + h) * head_dim;
                    for (i, &t) in valid_positions.iter().enumerate() {
                        let attn_w = (scores[i] - max_score).exp() * inv_sum;
                        let kv_base = (b * kv_seqlen + t) * head_dim;
                        for d in 0..head_dim {
                            let cur = half::bf16::from_bits(out_data[o_base + d]).to_f32();
                            out_data[o_base + d] = half::bf16::from_f32(
                                cur + attn_w * kv_bf16[kv_base + d].to_f32()
                            ).to_bits();
                        }
                    }
                }
            }
        }

        let out_cpu = CpuTensor::new(
            bytemuck::cast_slice(&out_data).to_vec(),
            vec![bsz, seqlen, n_heads, head_dim],
            DType::BF16,
        );
        GpuTensor::from_host(device, &out_cpu)
    }

    fn compute_topk_idxs(
        &self,
        bsz: usize,
        seqlen: usize,
        start_pos: usize,
    ) -> Result<GpuTensor> {
        let win = self.config.sliding_window;
        let device = self.kv_cache.cache.device.clone();

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

        if last_dim == 2048 {
            let y = GpuTensor::zeros(device.clone(), shape.clone(), DType::BF16)?;
            if self.kernels.call("swiglu_N2048", &[gate, up, &y]).is_ok() {
                return Ok(y);
            }
        }

        let limit = self.config.swiglu_limit as f32;
        let gate_host = gate.to_host()?;
        let up_host = up.to_host()?;
        let gate_bf16: &[half::bf16] = bytemuck::cast_slice(&gate_host.data);
        let up_bf16: &[half::bf16] = bytemuck::cast_slice(&up_host.data);

        let mut out_data = vec![0u16; gate_bf16.len()];
        for i in 0..gate_bf16.len() {
            let g = gate_bf16[i].to_f32();
            let u = up_bf16[i].to_f32();
            let g_f64 = g as f64;
            let u_f64 = u as f64;
            let limit_f64 = limit as f64;
            let g_clamped = if limit > 0.0 { g_f64.clamp(-1e10, limit_f64) } else { g_f64 };
            let u_clamped = if limit > 0.0 { u_f64.clamp(-limit_f64, limit_f64) } else { u_f64 };
            let silu = g_clamped / (1.0 + (-g_clamped).exp());
            out_data[i] = half::bf16::from_f32((silu * u_clamped) as f32).to_bits();
        }

        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out_data).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
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
            4096 if x.dtype == DType::BF16 => Some("cast_bf16_to_f32_N4096"),
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
            let mut y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::FP32)?;
            if self.kernels.call(kname, &[&x_2d, &mut y_2d]).is_ok() {
                let mut out_shape = x.shape.clone();
                out_shape.last_mut().unwrap();
                return Ok(GpuTensor {
                    slice: y_2d.slice,
                    shape: x.shape.clone(),
                    dtype: DType::FP32,
                    device,
                });
            }
        }

        let host = x.to_host()?;
        let out = match host.dtype {
            DType::BF16 => {
                let bf16_slice: &[half::bf16] = bytemuck::cast_slice(&host.data);
                let f32_data: Vec<f32> = bf16_slice.iter().map(|v| v.to_f32()).collect();
                CpuTensor::new(bytemuck::cast_slice(&f32_data).to_vec(), host.shape, DType::FP32)
            }
            _ => return Err(anyhow!("cast_to_f32: unsupported dtype {:?}", host.dtype)),
        };
        GpuTensor::from_host(x.device.clone(), &out)
    }

    fn cast_to_bf16(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype == DType::BF16 {
            return Ok(x.clone());
        }
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            4096 | 16384 if x.dtype == DType::FP32 => Some("cast_f32_to_bf16_N4096"),
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
            let mut y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::BF16)?;
            if self.kernels.call(kname, &[&x_2d, &mut y_2d]).is_ok() {
                return Ok(GpuTensor {
                    slice: y_2d.slice,
                    shape: x.shape.clone(),
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let host = x.to_host()?;
        let out = match host.dtype {
            DType::FP32 => {
                let f32_slice: &[f32] = bytemuck::cast_slice(&host.data);
                let bf16_data: Vec<half::bf16> = f32_slice.iter().map(|v| half::bf16::from_f32(*v)).collect();
                CpuTensor::new(bytemuck::cast_slice(&bf16_data).to_vec(), host.shape, DType::BF16)
            }
            _ => return Err(anyhow!("cast_to_bf16: unsupported dtype {:?}", host.dtype)),
        };
        GpuTensor::from_host(x.device.clone(), &out)
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
