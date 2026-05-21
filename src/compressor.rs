use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use std::sync::Arc;

pub struct Compressor {
    pub wkv: GpuTensor,
    pub wgate: GpuTensor,
    pub rope: RopeCache,
    pub compress_ratio: usize,
    pub overlap: bool,
    pub rotate: bool,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub kv_state: Vec<f32>,
    pub score_state: Vec<f32>,
    pub max_batch: usize,
    pub coff: usize,
    pub device: Arc<CudaContext>,
    pub cublas: Arc<CublasHandle>,
    ape_cpu: Vec<f32>,
    norm_cpu: Vec<f32>,
    norm_gpu: GpuTensor,
    hadamard_gpu: Option<GpuTensor>,
    ape_gpu: GpuTensor,
    kernels: Arc<KernelRegistry>,
    kv_state_gpu: Option<GpuTensor>,
    score_state_gpu: Option<GpuTensor>,
}

impl Compressor {
    pub fn new(
        device: Arc<CudaContext>,
        config: &ModelConfig,
        layer_id: usize,
        compress_ratio: usize,
        head_dim: usize,
        rotate: bool,
        max_batch: usize,
        loader: &mut WeightLoader,
        cublas: Arc<CublasHandle>,
        kernels: Arc<KernelRegistry>,
    ) -> Result<Self> {
        let p = format!("layers.{}.attn.compressor.", layer_id);

        let wkv_cpu = loader.load(&(p.clone() + "wkv.weight"))
            .with_context(|| format!("compressor: wkv.weight"))?;
        let wkv = Self::cpu_to_fp32_gpu(&wkv_cpu, device.clone())?;

        let wgate_cpu = loader.load(&(p.clone() + "wgate.weight"))
            .with_context(|| format!("compressor: wgate.weight"))?;
        let wgate = Self::cpu_to_fp32_gpu(&wgate_cpu, device.clone())?;

        let norm_cpu_tensor = loader.load(&(p.clone() + "norm.weight"))
            .with_context(|| format!("compressor: norm.weight"))?;
        let norm_cpu: Vec<f32> = match norm_cpu_tensor.dtype {
            DType::BF16 => {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&norm_cpu_tensor.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            }
            DType::FP32 => bytemuck::cast_slice(&norm_cpu_tensor.data).to_vec(),
            _ => anyhow::bail!("unsupported norm dtype"),
        };

        let ape_cpu_tensor = loader.load(&(p.clone() + "ape"))
            .with_context(|| format!("compressor: ape"))?;
        let ape_cpu: Vec<f32> = match ape_cpu_tensor.dtype {
            DType::BF16 => {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&ape_cpu_tensor.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            }
            DType::FP32 => bytemuck::cast_slice(&ape_cpu_tensor.data).to_vec(),
            _ => anyhow::bail!("unsupported ape dtype"),
        };

        let overlap = compress_ratio == 4;
        let coff = 1 + overlap as usize;
        let rope_head_dim = config.qk_rope_head_dim;

        let state_size = coff * compress_ratio * coff * head_dim;
        let kv_state = vec![0.0f32; max_batch * state_size];
        let score_state = vec![f32::NEG_INFINITY; max_batch * state_size];

        let kv_state_gpu = {
            let state_cpu = CpuTensor::new(vec![0u8; state_size * 4], vec![max_batch, state_size], DType::FP32);
            GpuTensor::from_host(device.clone(), &state_cpu).ok()
        };
        let score_state_gpu = {
            let state_cpu = CpuTensor::new(vec![0u8; state_size * 4], vec![max_batch, state_size], DType::FP32);
            GpuTensor::from_host(device.clone(), &state_cpu).ok()
        };

        let rope = RopeCache::precompute(config, config.max_position_embeddings, layer_id);

        let norm_gpu = {
            let nt = CpuTensor::new(
                bytemuck::cast_slice(&norm_cpu).to_vec(),
                vec![head_dim],
                DType::FP32,
            );
            GpuTensor::from_host(device.clone(), &nt)?
        };

        let hadamard_gpu = if rotate {
            let h = Self::precompute_hadamard_matrix(head_dim);
            let ht = CpuTensor::new(
                bytemuck::cast_slice(&h).to_vec(),
                vec![head_dim, head_dim],
                DType::FP32,
            );
            Some(GpuTensor::from_host(device.clone(), &ht)?)
        } else {
            None
        };

        let ape_gpu = {
            let od = coff * head_dim;
            let ape_rows = if od > 0 { ape_cpu.len() / od } else { 1 };
            let at = CpuTensor::new(
                bytemuck::cast_slice(&ape_cpu).to_vec(),
                vec![ape_rows, od],
                DType::FP32,
            );
            GpuTensor::from_host(device.clone(), &at)?
        };

        Ok(Self {
            wkv,
            wgate,
            rope,
            compress_ratio,
            overlap,
            rotate,
            head_dim,
            rope_head_dim,
            kv_state,
            score_state,
            max_batch,
            coff,
            device,
            cublas,
            ape_cpu,
            norm_cpu,
            norm_gpu,
            hadamard_gpu,
            ape_gpu,
            kernels,
            kv_state_gpu,
            score_state_gpu,
        })
    }

    pub fn precompute_hadamard_matrix(n: usize) -> Vec<f32> {
        let mut h = vec![0.0f32; n * n];
        h[0] = 1.0;
        let mut size = 1;
        while size < n {
            for i in 0..size {
                for j in 0..size {
                    let val = h[i * n + j];
                    h[(i + size) * n + j] = val;
                    h[i * n + (j + size)] = val;
                    h[(i + size) * n + (j + size)] = -val;
                }
            }
            size *= 2;
        }
        let scale = 1.0 / (n as f32).sqrt();
        for v in h.iter_mut() {
            *v *= scale;
        }
        h
    }

    fn try_gpu_postprocess(
        &self,
        pool_gpu: &GpuTensor,
        bsz: usize,
        n_groups: usize,
        start_pos: usize,
        rope_start_pos: usize,
    ) -> Option<GpuTensor> {
        let d = self.head_dim;
        let rd = self.rope_head_dim;
        let total = bsz * n_groups;
        let half_rd = rd / 2;

        let normed = GpuTensor::zeros(self.device.clone(), vec![total, d], DType::FP32).ok()?;
        let norm_kernel = format!("rmsnorm_f32_weighted_N{}", d);
        self.kernels.call(&norm_kernel, &[pool_gpu, &self.norm_gpu, &normed]).ok()?;

        let mut cos_data = Vec::with_capacity(total * half_rd);
        let mut sin_data = Vec::with_capacity(total * half_rd);
        for _b in 0..bsz {
            for g in 0..n_groups {
                let pos = if start_pos == 0 { g * self.compress_ratio } else { rope_start_pos };
                let (cos, sin) = self.rope.get_slice(pos, 1);
                cos_data.extend_from_slice(cos);
                sin_data.extend_from_slice(sin);
            }
        }
        let cos_cpu = CpuTensor::new(
            bytemuck::cast_slice(&cos_data).to_vec(),
            vec![total, half_rd],
            DType::FP32,
        );
        let sin_cpu = CpuTensor::new(
            bytemuck::cast_slice(&sin_data).to_vec(),
            vec![total, half_rd],
            DType::FP32,
        );
        let cos_gpu = GpuTensor::from_host(self.device.clone(), &cos_cpu).ok()?;
        let sin_gpu = GpuTensor::from_host(self.device.clone(), &sin_cpu).ok()?;

        let roped = GpuTensor::zeros(self.device.clone(), vec![total, d], DType::FP32).ok()?;
        let rope_kernel = format!("compressor_rope_f32_d{}_rd{}", d, rd);
        self.kernels.call(
            &rope_kernel,
            &[&normed, &cos_gpu, &sin_gpu, &roped],
        ).ok()?;

        let quantized = if self.rotate {
            let hadamard = self.hadamard_gpu.as_ref()?;
            let mut hadamarded = GpuTensor::zeros(self.device.clone(), vec![total, d], DType::FP32).ok()?;
            self.cublas.gemm_f32(total, d, d, &roped, hadamard, &mut hadamarded, 1.0, 0.0).ok()?;

            let qdq = GpuTensor::zeros(self.device.clone(), vec![total, d], DType::FP32).ok()?;
            let qdq_kernel = format!("fp4_qdq_f32_N{}_bs32", d);
            self.kernels.call(&qdq_kernel, &[&hadamarded, &qdq]).ok()?;
            qdq
        } else {
            let host = roped.to_host().ok()?;
            let mut kv: Vec<f32> = bytemuck::cast_slice(&host.data).to_vec();
            let nope_dim = d - rd;
            let fp8_max = 448.0f32;
            let block_size = 64usize;
            for b in 0..bsz {
                for g in 0..n_groups {
                    let off = (b * n_groups + g) * d;
                    for block_start in (0..nope_dim).step_by(block_size) {
                        let block_end = (block_start + block_size).min(nope_dim);
                        let mut amax: f32 = 0.0;
                        for dd in block_start..block_end {
                            amax = amax.max(kv[off + dd].abs());
                        }
                        amax = amax.max(1e-4);
                        let scale = amax / fp8_max;
                        for dd in block_start..block_end {
                            let q = kv[off + dd] / scale;
                            let fp8_bits = f32_to_fp8_e4m3(q);
                            let deq = fp8_e4m3_to_f32(fp8_bits);
                            kv[off + dd] = deq * scale;
                        }
                    }
                }
            }
            let kv_cpu = CpuTensor::new(
                bytemuck::cast_slice(&kv).to_vec(),
                vec![total, d],
                DType::FP32,
            );
            GpuTensor::from_host(self.device.clone(), &kv_cpu).ok()?
        };

        let result_bf16 = GpuTensor::zeros(self.device.clone(), vec![total, d], DType::BF16).ok()?;
        let cast_kernel = format!("cast_f32_to_bf16_N{}", d);
        self.kernels.call(&cast_kernel, &[&quantized, &result_bf16]).ok()?;

        Some(GpuTensor {
            slice: result_bf16.slice,
            shape: vec![bsz, n_groups, d],
            dtype: DType::BF16,
            device: result_bf16.device,
        })
    }

    fn cpu_postprocess_to_gpu(
        &self,
        result: &mut [f32],
        bsz: usize,
        n_groups: usize,
        start_pos: usize,
        rope_start_pos: usize,
    ) -> Result<GpuTensor> {
        let d = self.head_dim;
        self.apply_norm_rope_quant(result, bsz, n_groups, start_pos, rope_start_pos)?;
        let result_bf16: Vec<u16> = result.iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect();
        let cpu = CpuTensor::new(
            bytemuck::cast_slice(&result_bf16).to_vec(),
            vec![bsz, n_groups, d],
            DType::BF16,
        );
        GpuTensor::from_host(self.device.clone(), &cpu)
    }

    fn cpu_to_fp32_gpu(cpu: &CpuTensor, device: Arc<CudaContext>) -> Result<GpuTensor> {
        let f32_data: Vec<f32> = match cpu.dtype {
            DType::BF16 => {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&cpu.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            }
            DType::FP32 => bytemuck::cast_slice(&cpu.data).to_vec(),
            _ => anyhow::bail!("unsupported dtype for fp32 conversion"),
        };
        let fp32_cpu = CpuTensor::new(
            bytemuck::cast_slice(&f32_data).to_vec(),
            cpu.shape.clone(),
            DType::FP32,
        );
        GpuTensor::from_host(device, &fp32_cpu)
    }

    fn try_gpu_pool(
        &self,
        kv_data: &[f32],
        score_data: &[f32],
        pool_size: usize,
        n_groups: usize,
        bsz: usize,
    ) -> Option<GpuTensor> {
        let d = self.head_dim;
        let kernel_name = format!("compressor_pool_d{}_c{}", d, pool_size);

        let total = bsz * n_groups;
        let kv_cpu = CpuTensor::new(
            bytemuck::cast_slice(kv_data).to_vec(),
            vec![total, pool_size, d],
            DType::FP32,
        );
        let kv_gpu = GpuTensor::from_host(self.device.clone(), &kv_cpu).ok()?;

        let gate_cpu = CpuTensor::new(
            bytemuck::cast_slice(score_data).to_vec(),
            vec![total, pool_size, d],
            DType::FP32,
        );
        let gate_gpu = GpuTensor::from_host(self.device.clone(), &gate_cpu).ok()?;

        let out_gpu = GpuTensor::zeros(
            self.device.clone(),
            vec![total, d],
            DType::FP32,
        ).ok()?;

        self.kernels.call(&kernel_name, &[&kv_gpu, &gate_gpu, &out_gpu]).ok()?;

        Some(out_gpu)
    }

    fn try_gpu_pool_gpu(
        &self,
        kv_gpu: &GpuTensor,
        score_gpu: &GpuTensor,
        pool_size: usize,
        n_groups: usize,
        bsz: usize,
    ) -> Option<GpuTensor> {
        let d = self.head_dim;
        let kernel_name = format!("compressor_pool_d{}_c{}", d, pool_size);
        let total = bsz * n_groups;

        let out_gpu = GpuTensor::zeros(
            self.device.clone(),
            vec![total, d],
            DType::FP32,
        ).ok()?;

        self.kernels.call(&kernel_name, &[kv_gpu, score_gpu, &out_gpu]).ok()?;

        Some(out_gpu)
    }

    fn try_gpu_group(
        &self,
        src: &GpuTensor,
        bsz: usize,
        seqlen: usize,
        n_groups: usize,
        pool_size: usize,
        is_score: bool,
    ) -> Option<GpuTensor> {
        let d = self.head_dim;
        let coff = self.coff;
        let ratio = self.compress_ratio;
        let out_dim = coff * d;
        let overlap = self.overlap;

        let kernel_name = format!("compressor_group_d{}_od{}_ps{}", d, out_dim, pool_size);

        let mut row_idx = vec![0i32; bsz * n_groups * pool_size];
        let mut col_off = vec![0i32; pool_size];

        if overlap {
            for r in 0..ratio {
                col_off[r] = 0;
                col_off[ratio + r] = d as i32;
            }
            for b in 0..bsz {
                for g in 0..n_groups {
                    let base = (b * n_groups + g) * pool_size;
                    for r in 0..ratio {
                        if g > 0 {
                            row_idx[base + r] = (b * seqlen + (g - 1) * ratio + r) as i32;
                        } else {
                            row_idx[base + r] = -1;
                        }
                        row_idx[base + ratio + r] = (b * seqlen + g * ratio + r) as i32;
                    }
                }
            }
        } else {
            for r in 0..ratio {
                col_off[r] = 0;
            }
            for b in 0..bsz {
                for g in 0..n_groups {
                    let base = (b * n_groups + g) * pool_size;
                    for r in 0..ratio {
                        row_idx[base + r] = (b * seqlen + g * ratio + r) as i32;
                    }
                }
            }
        }

        let total = bsz * n_groups;
        let ri_cpu = CpuTensor::new(
            bytemuck::cast_slice(&row_idx).to_vec(),
            vec![total, pool_size],
            DType::INT32,
        );
        let ri_gpu = GpuTensor::from_host(self.device.clone(), &ri_cpu).ok()?;

        let co_cpu = CpuTensor::new(
            bytemuck::cast_slice(&col_off).to_vec(),
            vec![pool_size],
            DType::INT32,
        );
        let co_gpu = GpuTensor::from_host(self.device.clone(), &co_cpu).ok()?;

        let is_score_val = if is_score { 1i32 } else { 0i32 };
        let is_cpu = CpuTensor::new(
            bytemuck::cast_slice(&[is_score_val]).to_vec(),
            vec![1],
            DType::INT32,
        );
        let is_gpu = GpuTensor::from_host(self.device.clone(), &is_cpu).ok()?;

        let dst = GpuTensor::zeros(
            self.device.clone(),
            vec![total, pool_size, d],
            DType::FP32,
        ).ok()?;

        self.kernels.call(
            &kernel_name,
            &[src, &self.ape_gpu, &ri_gpu, &co_gpu, &is_gpu, &dst],
        ).ok()?;

        Some(dst)
    }

    fn x_to_fp32_gpu(&self, x: &GpuTensor, total: usize, dim: usize) -> Result<GpuTensor> {
        let x_2d = GpuTensor {
            slice: x.slice.clone(),
            shape: vec![total, dim],
            dtype: x.dtype,
            device: x.device.clone(),
        };

        if x.dtype == DType::BF16 {
            let kernel_name = match dim {
                4096 => Some("cast_bf16_to_f32_N4096"),
                16384 => Some("cast_bf16_to_f32_N16384"),
                _ => None,
            };
            if let Some(kname) = kernel_name {
                let x_f32 = GpuTensor::zeros(self.device.clone(), vec![total, dim], DType::FP32)?;
                if self.kernels.call(kname, &[&x_2d, &x_f32]).is_ok() {
                    return Ok(x_f32);
                }
            }
        }

        let x_host = x_2d.to_host()?;
        let x_f32: Vec<f32> = match x.dtype {
            DType::BF16 => {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            }
            DType::FP32 => bytemuck::cast_slice(&x_host.data).to_vec(),
            _ => anyhow::bail!("unsupported dtype for fp32 conversion"),
        };
        let x_fp32_cpu = CpuTensor::new(
            bytemuck::cast_slice(&x_f32).to_vec(),
            vec![total, dim],
            DType::FP32,
        );
        GpuTensor::from_host(self.device.clone(), &x_fp32_cpu)
    }

    fn try_gpu_decode(
        &mut self,
        kv_gpu: &GpuTensor,
        score_gpu: &GpuTensor,
        bsz: usize,
        start_pos: usize,
    ) -> Option<GpuTensor> {
        let ratio = self.compress_ratio;
        let d = self.head_dim;
        let coff = self.coff;
        let overlap = self.overlap;
        let pos_in_ratio = start_pos % ratio;
        let should_compress = (start_pos + 1) % ratio == 0;
        let out_dim = coff * d;

        let ape_off = pos_in_ratio * coff * d;
        let ape_nonzero = ape_off < self.ape_cpu.len() && self.ape_cpu[ape_off..ape_off + out_dim.min(self.ape_cpu.len().saturating_sub(ape_off))].iter().any(|&v| v != 0.0);
        if ape_nonzero {
            return None;
        }

        {
            let kv_state = self.kv_state_gpu.as_mut()?;
            let score_state = self.score_state_gpu.as_mut()?;

            let state_off = pos_in_ratio * coff * d;
            let dst_offset = state_off * 4;

            if overlap {
                let overlap_off = (ratio + pos_in_ratio) * coff * d;
                let overlap_dst = overlap_off * 4;

                GpuTensor::d2d_scatter_rows(
                    kv_gpu,
                    kv_state,
                    out_dim * 4,
                    coff * ratio * coff * d * 4,
                    out_dim * 4,
                    overlap_dst,
                    bsz,
                ).ok()?;
                GpuTensor::d2d_scatter_rows(
                    score_gpu,
                    score_state,
                    out_dim * 4,
                    coff * ratio * coff * d * 4,
                    out_dim * 4,
                    overlap_dst,
                    bsz,
                ).ok()?;
            } else {
                GpuTensor::d2d_scatter_rows(
                    kv_gpu,
                    kv_state,
                    out_dim * 4,
                    coff * ratio * coff * d * 4,
                    out_dim * 4,
                    dst_offset,
                    bsz,
                ).ok()?;
                GpuTensor::d2d_scatter_rows(
                    score_gpu,
                    score_state,
                    out_dim * 4,
                    coff * ratio * coff * d * 4,
                    out_dim * 4,
                    dst_offset,
                    bsz,
                ).ok()?;
            }
        }

        if !should_compress {
            return None;
        }

        let pool_size = if overlap { coff * ratio } else { ratio };
        let n_groups = 1usize;

        let kv_state = self.kv_state_gpu.as_ref()?;
        let score_state = self.score_state_gpu.as_ref()?;

        let pooled = self.try_gpu_pool_gpu(kv_state, score_state, pool_size, n_groups, bsz)?;

        if overlap {
            if let Some(ref mut kv_state_gpu) = self.kv_state_gpu {
                for r in 0..ratio {
                    let src_off = (ratio + r) * coff * d * 4;
                    let dst_off = r * coff * d * 4;
                    let copy_bytes = coff * d * 4;
                    let _ = GpuTensor::d2d_copy_within(kv_state_gpu, src_off, dst_off, copy_bytes);
                }
            }
            if let Some(ref mut score_state_gpu) = self.score_state_gpu {
                for r in 0..ratio {
                    let src_off = (ratio + r) * coff * d * 4;
                    let dst_off = r * coff * d * 4;
                    let copy_bytes = coff * d * 4;
                    let _ = GpuTensor::d2d_copy_within(score_state_gpu, src_off, dst_off, copy_bytes);
                }
            }
        }

        self.try_gpu_postprocess(&pooled, bsz, n_groups, start_pos, start_pos + 1 - ratio)
    }

    fn sync_cpu_state_from_gpu(&mut self) {
        if let Some(ref kv_state_gpu) = self.kv_state_gpu {
            if let Ok(host) = kv_state_gpu.to_host() {
                let f32_data: &[f32] = bytemuck::cast_slice(&host.data);
                let len = f32_data.len().min(self.kv_state.len());
                for i in 0..len {
                    self.kv_state[i] = f32_data[i];
                }
            }
        }
        if let Some(ref score_state_gpu) = self.score_state_gpu {
            if let Ok(host) = score_state_gpu.to_host() {
                let f32_data: &[f32] = bytemuck::cast_slice(&host.data);
                let len = f32_data.len().min(self.score_state.len());
                for i in 0..len {
                    self.score_state[i] = f32_data[i];
                }
            }
        }
    }

    pub fn forward(
        &mut self,
        x: &GpuTensor,
        start_pos: usize,
        bsz: usize,
        seqlen: usize,
    ) -> Result<Option<GpuTensor>> {
        let ratio = self.compress_ratio;
        let overlap = self.overlap;
        let d = self.head_dim;
        let coff = self.coff;
        let out_dim = coff * d;
        let total = bsz * seqlen;
        let dim = x.shape[x.shape.len() - 1];

        let x_fp32 = self.x_to_fp32_gpu(x, total, dim)?;

        let mut kv_gpu = GpuTensor::zeros(self.device.clone(), vec![total, out_dim], DType::FP32)?;
        self.cublas.gemm_f32(total, out_dim, dim, &x_fp32, &self.wkv, &mut kv_gpu, 1.0, 0.0)?;

        let mut score_gpu = GpuTensor::zeros(self.device.clone(), vec![total, out_dim], DType::FP32)?;
        self.cublas.gemm_f32(total, out_dim, dim, &x_fp32, &self.wgate, &mut score_gpu, 1.0, 0.0)?;

        let mut compressed = None;

        if start_pos == 0 {
            let should_compress = seqlen >= ratio;
            if !should_compress {
                return Ok(None);
            }

            let remainder = seqlen % ratio;
            let cutoff = seqlen - remainder;
            let _offset = if overlap { ratio } else { 0 };
            let n_groups = cutoff / ratio;
            let pool_size = if overlap { 2 * ratio } else { ratio };

            let gpu_group_kv = self.try_gpu_group(&kv_gpu, bsz, seqlen, n_groups, pool_size, false);
            let gpu_group_score = self.try_gpu_group(&score_gpu, bsz, seqlen, n_groups, pool_size, true);

            if let (Some(gkv), Some(gscore)) = (&gpu_group_kv, &gpu_group_score) {
                let result_gpu = self.try_gpu_pool_gpu(gkv, gscore, pool_size, n_groups, bsz);

                if let Some(ref gpu) = result_gpu {
                    if let Some(gpu_out) = self.try_gpu_postprocess(gpu, bsz, n_groups, start_pos, cutoff) {
                        compressed = Some(gpu_out);
                    } else {
                        let host = gpu.to_host()?;
                        let mut result: Vec<f32> = bytemuck::cast_slice(&host.data).to_vec();
                        compressed = Some(self.cpu_postprocess_to_gpu(&mut result, bsz, n_groups, start_pos, cutoff)?);
                    }
                } else {
                    let gkv_host = gkv.to_host()?;
                    let kv_grouped: Vec<f32> = bytemuck::cast_slice(&gkv_host.data).to_vec();
                    let gscore_host = gscore.to_host()?;
                    let score_grouped: Vec<f32> = bytemuck::cast_slice(&gscore_host.data).to_vec();

                    let mut result = vec![0.0f32; bsz * n_groups * d];
                    for b in 0..bsz {
                        for g in 0..n_groups {
                            for dd in 0..d {
                                let mut max_s = f32::NEG_INFINITY;
                                for r in 0..pool_size {
                                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    max_s = max_s.max(score_grouped[idx]);
                                }
                                let mut sum_exp = 0.0f32;
                                let mut weights = vec![0.0f32; pool_size];
                                for r in 0..pool_size {
                                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    weights[r] = (score_grouped[idx] - max_s).exp();
                                    sum_exp += weights[r];
                                }
                                if sum_exp > 0.0 {
                                    for w in &mut weights { *w /= sum_exp; }
                                }
                                let res_off = (b * n_groups + g) * d + dd;
                                for r in 0..pool_size {
                                    let kv_idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    result[res_off] += weights[r] * kv_grouped[kv_idx];
                                }
                            }
                        }
                    }
                    compressed = Some(self.cpu_postprocess_to_gpu(&mut result, bsz, n_groups, start_pos, cutoff)?);
                }
            } else {
                let kv_host = kv_gpu.to_host()?;
                let kv_proj: Vec<f32> = bytemuck::cast_slice(&kv_host.data).to_vec();
                let score_host = score_gpu.to_host()?;
                let score_proj: Vec<f32> = bytemuck::cast_slice(&score_host.data).to_vec();
                let ape_f32 = &self.ape_cpu;

                if overlap && cutoff >= ratio {
                    for b in 0..bsz {
                        for r in 0..ratio {
                            let src_t = b * seqlen + (cutoff - ratio + r);
                            let dst_off = b * coff * ratio * coff * d + r * coff * d;
                            for dd in 0..coff * d {
                                self.kv_state[dst_off + dd] = kv_proj[src_t * out_dim + dd];
                            }
                            let ape_off = r * coff * d;
                            for dd in 0..coff * d {
                                let ape_val = if ape_off + dd < ape_f32.len() { ape_f32[ape_off + dd] } else { 0.0 };
                                self.score_state[dst_off + dd] = score_proj[src_t * out_dim + dd] + ape_val;
                            }
                        }
                    }
                }

                let mut kv_main = Vec::new();
                let mut score_main = Vec::new();
                if remainder > 0 {
                    for b in 0..bsz {
                        for t in 0..cutoff {
                            let src_t = b * seqlen + t;
                            for dd in 0..out_dim {
                                kv_main.push(kv_proj[src_t * out_dim + dd]);
                                score_main.push(score_proj[src_t * out_dim + dd]);
                            }
                        }
                        for r in 0..remainder {
                            let src_t = b * seqlen + cutoff + r;
                            let dst_off = b * coff * ratio * coff * d + (_offset + r) * coff * d;
                            for dd in 0..coff * d {
                                self.kv_state[dst_off + dd] = kv_proj[src_t * out_dim + dd];
                            }
                            let ape_off = r * coff * d;
                            for dd in 0..coff * d {
                                let ape_val = if ape_off + dd < ape_f32.len() { ape_f32[ape_off + dd] } else { 0.0 };
                                self.score_state[dst_off + dd] = score_proj[src_t * out_dim + dd] + ape_val;
                            }
                        }
                    }
                } else {
                    for b in 0..bsz {
                        for t in 0..cutoff {
                            let src_t = b * seqlen + t;
                            for dd in 0..out_dim {
                                kv_main.push(kv_proj[src_t * out_dim + dd]);
                                score_main.push(score_proj[src_t * out_dim + dd]);
                            }
                        }
                    }
                }

                let mut kv_grouped = vec![0.0f32; bsz * n_groups * ratio * coff * d];
                let mut score_grouped = vec![0.0f32; bsz * n_groups * ratio * coff * d];

                for b in 0..bsz {
                    for g in 0..n_groups {
                        for r in 0..ratio {
                            let src_off = (b * cutoff + g * ratio + r) * out_dim;
                            let dst_off = (b * n_groups + g) * ratio * coff * d + r * coff * d;
                            for dd in 0..coff * d {
                                kv_grouped[dst_off + dd] = kv_main[src_off + dd];
                                let ape_off = r * coff * d;
                                let ape_val = if ape_off + dd < ape_f32.len() { ape_f32[ape_off + dd] } else { 0.0 };
                                score_grouped[dst_off + dd] = score_main[src_off + dd] + ape_val;
                            }
                        }
                    }
                }

                if overlap {
                    let mut kv_overlap = vec![0.0f32; bsz * n_groups * 2 * ratio * d];
                    let mut score_overlap = vec![f32::NEG_INFINITY; bsz * n_groups * 2 * ratio * d];

                    for b in 0..bsz {
                        for g in 0..n_groups {
                            for r in 0..ratio {
                                let src_off = (b * n_groups + g) * ratio * coff * d + r * coff * d;
                                let dst_off = (b * n_groups + g) * 2 * ratio * d + (ratio + r) * d;
                                for dd in 0..d {
                                    kv_overlap[dst_off + dd] = kv_grouped[src_off + d + dd];
                                    score_overlap[dst_off + dd] = score_grouped[src_off + d + dd];
                                }
                            }
                            if g > 0 {
                                for r in 0..ratio {
                                    let src_off = (b * n_groups + (g - 1)) * ratio * coff * d + r * coff * d;
                                    let dst_off = (b * n_groups + g) * 2 * ratio * d + r * d;
                                    for dd in 0..d {
                                        kv_overlap[dst_off + dd] = kv_grouped[src_off + dd];
                                        score_overlap[dst_off + dd] = score_grouped[src_off + dd];
                                    }
                                }
                            }
                        }
                    }

                    kv_grouped = kv_overlap;
                    score_grouped = score_overlap;
                }

                let result_gpu = self.try_gpu_pool(&kv_grouped, &score_grouped, pool_size, n_groups, bsz);

                if let Some(ref gpu) = result_gpu {
                    if let Some(gpu_out) = self.try_gpu_postprocess(gpu, bsz, n_groups, start_pos, cutoff) {
                        compressed = Some(gpu_out);
                    } else {
                        let host = gpu.to_host()?;
                        let mut result: Vec<f32> = bytemuck::cast_slice(&host.data).to_vec();
                        compressed = Some(self.cpu_postprocess_to_gpu(&mut result, bsz, n_groups, start_pos, cutoff)?);
                    }
                } else {
                    let mut result = vec![0.0f32; bsz * n_groups * d];
                    for b in 0..bsz {
                        for g in 0..n_groups {
                            for dd in 0..d {
                                let mut max_s = f32::NEG_INFINITY;
                                for r in 0..pool_size {
                                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    max_s = max_s.max(score_grouped[idx]);
                                }
                                let mut sum_exp = 0.0f32;
                                let mut weights = vec![0.0f32; pool_size];
                                for r in 0..pool_size {
                                    let idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    weights[r] = (score_grouped[idx] - max_s).exp();
                                    sum_exp += weights[r];
                                }
                                if sum_exp > 0.0 {
                                    for w in &mut weights { *w /= sum_exp; }
                                }
                                let res_off = (b * n_groups + g) * d + dd;
                                for r in 0..pool_size {
                                    let kv_idx = (b * n_groups + g) * pool_size * d + r * d + dd;
                                    result[res_off] += weights[r] * kv_grouped[kv_idx];
                                }
                            }
                        }
                    }
                    compressed = Some(self.cpu_postprocess_to_gpu(&mut result, bsz, n_groups, start_pos, cutoff)?);
                }
            }

            if let Some(ref mut kv_state_gpu) = self.kv_state_gpu {
                let state_cpu = CpuTensor::new(
                    bytemuck::cast_slice(&self.kv_state).to_vec(),
                    kv_state_gpu.shape.clone(),
                    DType::FP32,
                );
                if let Ok(new_gpu) = GpuTensor::from_host(self.device.clone(), &state_cpu) {
                    *kv_state_gpu = new_gpu;
                }
            }
            if let Some(ref mut score_state_gpu) = self.score_state_gpu {
                let state_cpu = CpuTensor::new(
                    bytemuck::cast_slice(&self.score_state).to_vec(),
                    score_state_gpu.shape.clone(),
                    DType::FP32,
                );
                if let Ok(new_gpu) = GpuTensor::from_host(self.device.clone(), &state_cpu) {
                    *score_state_gpu = new_gpu;
                }
            }
        } else {
            if let Some(gpu_result) = self.try_gpu_decode(&kv_gpu, &score_gpu, bsz, start_pos) {
                self.sync_cpu_state_from_gpu();
                return Ok(Some(gpu_result));
            }

            let kv_host = kv_gpu.to_host()?;
            let kv_proj: Vec<f32> = bytemuck::cast_slice(&kv_host.data).to_vec();
            let score_host = score_gpu.to_host()?;
            let score_proj: Vec<f32> = bytemuck::cast_slice(&score_host.data).to_vec();
            let ape_f32 = &self.ape_cpu;

            let should_compress = (start_pos + 1) % ratio == 0;
            let pos_in_ratio = start_pos % ratio;

            for b in 0..bsz {
                let src_t = b;
                let state_off = b * coff * ratio * coff * d;
                let kv_off = src_t * out_dim;
                let score_off = src_t * out_dim;

                let ape_off = pos_in_ratio * coff * d;
                for dd in 0..out_dim {
                    let ape_val = if ape_off + dd < ape_f32.len() { ape_f32[ape_off + dd] } else { 0.0 };
                    let s = score_proj[score_off + dd] + ape_val;

                    if overlap {
                        let state_idx = state_off + (ratio + pos_in_ratio) * coff * d + dd;
                        if state_idx < self.kv_state.len() {
                            self.kv_state[state_idx] = kv_proj[kv_off + dd];
                        }
                        if state_idx < self.score_state.len() {
                            self.score_state[state_idx] = s;
                        }
                    } else {
                        let state_idx = state_off + pos_in_ratio * coff * d + dd;
                        if state_idx < self.kv_state.len() {
                            self.kv_state[state_idx] = kv_proj[kv_off + dd];
                        }
                        if state_idx < self.score_state.len() {
                            self.score_state[state_idx] = s;
                        }
                    }
                }

                if should_compress {
                    let mut comp_result = vec![0.0f32; d];

                    if overlap {
                        let pool_size = coff * ratio;
                        let mut combined_kv = vec![0.0f32; pool_size * d];
                        let mut combined_score = vec![0.0f32; pool_size * d];

                        for r in 0..ratio {
                            let src_off = state_off + r * coff * d;
                            for dd in 0..d {
                                combined_kv[r * d + dd] = self.kv_state[src_off + dd];
                                combined_score[r * d + dd] = self.score_state[src_off + dd];
                            }
                        }
                        for r in 0..ratio {
                            let src_off = state_off + (ratio + r) * coff * d + d;
                            for dd in 0..d {
                                combined_kv[(ratio + r) * d + dd] = self.kv_state[src_off + dd];
                                combined_score[(ratio + r) * d + dd] = self.score_state[src_off + dd];
                            }
                        }

                        for dd in 0..d {
                            let mut max_s = f32::NEG_INFINITY;
                            for r in 0..pool_size {
                                let idx = r * d + dd;
                                max_s = max_s.max(combined_score[idx]);
                            }
                            let mut sum_exp = 0.0f32;
                            let mut weights = vec![0.0f32; pool_size];
                            for r in 0..pool_size {
                                let idx = r * d + dd;
                                weights[r] = (combined_score[idx] - max_s).exp();
                                sum_exp += weights[r];
                            }
                            if sum_exp > 0.0 {
                                for w in &mut weights { *w /= sum_exp; }
                            }
                            for r in 0..pool_size {
                                comp_result[dd] += weights[r] * combined_kv[r * d + dd];
                            }
                        }

                        for r in 0..ratio {
                            let dst_off = state_off + r * coff * d;
                            let src_off = state_off + (ratio + r) * coff * d;
                            for dd in 0..coff * d {
                                if dst_off + dd < self.kv_state.len() && src_off + dd < self.kv_state.len() {
                                    self.kv_state[dst_off + dd] = self.kv_state[src_off + dd];
                                }
                                if dst_off + dd < self.score_state.len() && src_off + dd < self.score_state.len() {
                                    self.score_state[dst_off + dd] = self.score_state[src_off + dd];
                                }
                            }
                        }
                    } else {
                        let _pool_size = ratio;
                        for dd in 0..d {
                            let mut max_s = f32::NEG_INFINITY;
                            for r in 0..ratio {
                                let score_off = state_off + r * coff * d + dd;
                                if score_off < self.score_state.len() {
                                    max_s = max_s.max(self.score_state[score_off]);
                                }
                            }
                            let mut sum_exp = 0.0f32;
                            let mut weights = vec![0.0f32; ratio];
                            for r in 0..ratio {
                                let score_off = state_off + r * coff * d + dd;
                                if score_off < self.score_state.len() {
                                    weights[r] = (self.score_state[score_off] - max_s).exp();
                                    sum_exp += weights[r];
                                }
                            }
                            if sum_exp > 0.0 {
                                for w in &mut weights { *w /= sum_exp; }
                            }
                            for r in 0..ratio {
                                let kv_off = state_off + r * coff * d + dd;
                                if kv_off < self.kv_state.len() {
                                    comp_result[dd] += weights[r] * self.kv_state[kv_off];
                                }
                            }
                        }
                    }

                    let comp_cpu = CpuTensor::new(
                        bytemuck::cast_slice(&comp_result).to_vec(),
                        vec![1, d],
                        DType::FP32,
                    );
                    let comp_gpu = GpuTensor::from_host(self.device.clone(), &comp_cpu)?;

                    if let Some(gpu_out) = self.try_gpu_postprocess(&comp_gpu, 1, 1, start_pos, start_pos + 1 - ratio) {
                        compressed = Some(gpu_out);
                    } else {
                        self.apply_norm_rope_quant(&mut comp_result, 1, 1, start_pos, start_pos + 1 - ratio)?;
                        let result_bf16: Vec<u16> = comp_result.iter()
                            .map(|v| half::bf16::from_f32(*v).to_bits())
                            .collect();
                        let cpu = CpuTensor::new(
                            bytemuck::cast_slice(&result_bf16).to_vec(),
                            vec![1, 1, d],
                            DType::BF16,
                        );
                        compressed = Some(GpuTensor::from_host(self.device.clone(), &cpu)?);
                    }
                }
            }

            if let Some(ref mut kv_state_gpu) = self.kv_state_gpu {
                let state_cpu = CpuTensor::new(
                    bytemuck::cast_slice(&self.kv_state).to_vec(),
                    kv_state_gpu.shape.clone(),
                    DType::FP32,
                );
                if let Ok(new_gpu) = GpuTensor::from_host(self.device.clone(), &state_cpu) {
                    *kv_state_gpu = new_gpu;
                }
            }
            if let Some(ref mut score_state_gpu) = self.score_state_gpu {
                let state_cpu = CpuTensor::new(
                    bytemuck::cast_slice(&self.score_state).to_vec(),
                    score_state_gpu.shape.clone(),
                    DType::FP32,
                );
                if let Ok(new_gpu) = GpuTensor::from_host(self.device.clone(), &state_cpu) {
                    *score_state_gpu = new_gpu;
                }
            }
        }

        Ok(compressed)
    }

    fn apply_norm_rope_quant(
        &self,
        kv: &mut [f32],
        bsz: usize,
        n_groups: usize,
        start_pos: usize,
        rope_start_pos: usize,
    ) -> Result<()> {
        let d = self.head_dim;
        let rd = self.rope_head_dim;
        let eps = 1e-6f32;
        let norm_f32 = &self.norm_cpu;

        for b in 0..bsz {
            for g in 0..n_groups {
                let off = (b * n_groups + g) * d;
                let sq_sum: f32 = (0..d).map(|dd| kv[off + dd].powi(2)).sum();
                let inv_norm = 1.0 / (sq_sum / d as f32 + eps).sqrt();
                for dd in 0..d {
                    let w = if dd < norm_f32.len() { norm_f32[dd] } else { 1.0 };
                    kv[off + dd] = kv[off + dd] * inv_norm * w;
                }

                let pos = if start_pos == 0 { g * self.compress_ratio } else { rope_start_pos };
                let (cos_data, sin_data) = self.rope.get_slice(pos, 1);
                let half_rd = rd / 2;
                let rope_start = d - rd;
                for k in 0..half_rd {
                    let idx1 = rope_start + 2 * k;
                    let idx2 = rope_start + 2 * k + 1;
                    let c = cos_data[k] as f64;
                    let s = sin_data[k] as f64;
                    let v1 = kv[off + idx1] as f64;
                    let v2 = kv[off + idx2] as f64;
                    kv[off + idx1] = (v1 * c - v2 * s) as f32;
                    kv[off + idx2] = (v1 * s + v2 * c) as f32;
                }

                if self.rotate {
                    crate::quant::hadamard_transform(&mut kv[off..off + d], d);
                    crate::quant::fp4_act_quant_qdq(&mut kv[off..off + d], d, 32);
                } else {
                    let block_size = 64usize;
                    let nope_dim = d - rd;
                    let fp8_max = 448.0f32;
                    for block_start in (0..nope_dim).step_by(block_size) {
                        let block_end = (block_start + block_size).min(nope_dim);
                        let mut amax: f32 = 0.0;
                        for dd in block_start..block_end {
                            amax = amax.max(kv[off + dd].abs());
                        }
                        amax = amax.max(1e-4);
                        let scale = amax / fp8_max;
                        for dd in block_start..block_end {
                            let q = kv[off + dd] / scale;
                            let fp8_bits = f32_to_fp8_e4m3(q);
                            let deq = fp8_e4m3_to_f32(fp8_bits);
                            kv[off + dd] = deq * scale;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
