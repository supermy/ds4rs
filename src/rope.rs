use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::tensor::{CpuTensor, GpuTensor};
use anyhow::Result;
use cudarc::driver::{CudaContext, DevicePtr};
use std::sync::Arc;

pub struct RopeCache {
    pub freqs_cos: Vec<f32>,
    pub freqs_sin: Vec<f32>,
    pub seqlen: usize,
    pub dim: usize,
    cos_gpu: Option<GpuTensor>,
    sin_gpu: Option<GpuTensor>,
    device: Option<Arc<CudaContext>>,
}

impl RopeCache {
    pub fn precompute(config: &ModelConfig, seqlen: usize, layer_id: usize) -> Self {
        let dim = config.qk_rope_head_dim;
        let base = config.rope_theta_for_layer(layer_id);
        let factor = config.rope_factor_for_layer(layer_id);
        let original_seq_len = config.original_seq_len_for_layer(layer_id);
        let beta_fast = config.beta_fast();
        let beta_slow = config.beta_slow();

        let half_dim = dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);

        for k in 0..half_dim {
            let f = 1.0 / base.powf(2.0 * k as f64 / dim as f64);
            freqs.push(f);
        }

        if original_seq_len > 0 && factor > 1.0 {
            let low = find_correction_dim(beta_fast as f64, dim as f64, base, original_seq_len as f64);
            let high = find_correction_dim(beta_slow as f64, dim as f64, base, original_seq_len as f64);
            let low = low.floor().max(0.0) as usize;
            let high = high.ceil().min((dim - 1) as f64) as usize;

            for k in 0..half_dim {
                let smooth = if low == high {
                    if k <= low { 1.0 } else { 0.0 }
                } else {
                    let linear = (k as f64 - low as f64) / (high as f64 - low as f64);
                    1.0 - linear.clamp(0.0, 1.0)
                };
                freqs[k] = freqs[k] / factor * (1.0 - smooth) + freqs[k] * smooth;
            }
        }

        let mut cos_data = Vec::with_capacity(seqlen * half_dim);
        let mut sin_data = Vec::with_capacity(seqlen * half_dim);

        for t in 0..seqlen {
            for k in 0..half_dim {
                let angle = t as f64 * freqs[k];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        Self {
            freqs_cos: cos_data,
            freqs_sin: sin_data,
            seqlen,
            dim,
            cos_gpu: None,
            sin_gpu: None,
            device: None,
        }
    }

    pub fn upload_to_gpu(&mut self, device: Arc<CudaContext>) -> Result<()> {
        let half_dim = self.dim / 2;
        let cos_cpu = CpuTensor::new(
            bytemuck::cast_slice(&self.freqs_cos).to_vec(),
            vec![self.seqlen, half_dim],
            DType::FP32,
        );
        let sin_cpu = CpuTensor::new(
            bytemuck::cast_slice(&self.freqs_sin).to_vec(),
            vec![self.seqlen, half_dim],
            DType::FP32,
        );
        self.cos_gpu = Some(GpuTensor::from_host(device.clone(), &cos_cpu)?);
        self.sin_gpu = Some(GpuTensor::from_host(device.clone(), &sin_cpu)?);
        self.device = Some(device);
        Ok(())
    }

    pub fn get_slice(&self, start_pos: usize, len: usize) -> (&[f32], &[f32]) {
        let half_dim = self.dim / 2;
        let cos_start = start_pos * half_dim;
        let sin_start = start_pos * half_dim;
        let cos_end = cos_start + len * half_dim;
        let sin_end = sin_start + len * half_dim;
        (
            &self.freqs_cos[cos_start..cos_end],
            &self.freqs_sin[sin_start..sin_end],
        )
    }

    pub fn get_gpu_slice(&self, start_pos: usize, len: usize) -> Option<(GpuTensor, GpuTensor)> {
        let cos_gpu = self.cos_gpu.as_ref()?;
        let sin_gpu = self.sin_gpu.as_ref()?;
        let half_dim = self.dim / 2;
        let device = self.device.as_ref()?;

        if start_pos == 0 {
            let cos = GpuTensor {
                slice: cos_gpu.slice.clone(),
                shape: vec![len, half_dim],
                dtype: DType::FP32,
                device: device.clone(),
            };
            let sin = GpuTensor {
                slice: sin_gpu.slice.clone(),
                shape: vec![len, half_dim],
                dtype: DType::FP32,
                device: device.clone(),
            };
            return Some((cos, sin));
        }

        let cos_sub = Self::d2d_extract_rows(cos_gpu, start_pos, len, half_dim, device.clone())?;
        let sin_sub = Self::d2d_extract_rows(sin_gpu, start_pos, len, half_dim, device.clone())?;
        Some((cos_sub, sin_sub))
    }

    fn d2d_extract_rows(
        src: &GpuTensor,
        row_start: usize,
        n_rows: usize,
        cols: usize,
        device: Arc<CudaContext>,
    ) -> Option<GpuTensor> {
        let elem_size = 4usize;
        let src_row_bytes = cols * elem_size;
        let src_offset = row_start * src_row_bytes;

        let out = GpuTensor::zeros(device.clone(), vec![n_rows, cols], DType::FP32).ok()?;
        let stream = device.default_stream();
        let (src_ptr, _src_guard) = src.slice.device_ptr(&stream);
        {
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            for r in 0..n_rows {
                unsafe {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + (r * src_row_bytes) as u64,
                        src_ptr + (src_offset + r * src_row_bytes) as u64,
                        src_row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
        }
        stream.synchronize().ok()?;
        Some(out)
    }
}

fn find_correction_dim(num_rotations: f64, dim: f64, base: f64, max_seq_len: f64) -> f64 {
    dim * (max_seq_len / (num_rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
}
