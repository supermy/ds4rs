use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::layer::TransformerLayer;
use crate::rope::RopeCache;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use crate::weight::WeightLoader;
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaContext, DevicePtr};
use std::sync::Arc;

pub struct Transformer {
    pub config: Arc<ModelConfig>,
    pub device: Arc<CudaContext>,
    pub layers: Vec<TransformerLayer>,
    pub embed: GpuTensor,
    pub head_weight: GpuTensor,
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
        let mut loader = WeightLoader::from_dir(model_dir)?;
        let cublas = Arc::new(CublasHandle::new(device.clone())?);

        let embed_cpu = loader.load("embed.weight").context("embed.weight")?;
        let embed = GpuTensor::from_host(device.clone(), &embed_cpu).context("embed upload")?;

        let head_cpu = loader.load("head.weight").context("head.weight")?;
        let head_weight = GpuTensor::from_host(device.clone(), &head_cpu).context("head upload")?;

        let head_scale = if loader.contains("head.scale") {
            let s = loader.load("head.scale")?;
            Some(GpuTensor::from_host(device.clone(), &s)?)
        } else {
            None
        };

        let norm_cpu = loader.load("norm.weight").context("norm.weight")?;
        let norm_weight = GpuTensor::from_host(device.clone(), &norm_cpu).context("norm upload")?;

        let hc_head_fn_cpu = loader.load("hc_head_fn").context("hc_head_fn")?;
        let hc_head_fn = GpuTensor::from_host(device.clone(), &hc_head_fn_cpu)?;

        let hc_head_scale_cpu = loader.load("hc_head_scale").context("hc_head_scale")?;
        let hc_head_scale = GpuTensor::from_host(device.clone(), &hc_head_scale_cpu)?;

        let hc_head_base_cpu = loader.load("hc_head_base").context("hc_head_base")?;
        let hc_head_base = GpuTensor::from_host(device.clone(), &hc_head_base_cpu)?;

        let mut ropes = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            ropes.push(RopeCache::precompute(&config, max_seqlen, layer_id));
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
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
        }

        Ok(Self {
            config,
            device,
            layers,
            embed,
            head_weight,
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

        let h = self.hc_expand(&h, hc)?;

        let mut h = h;
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, start_pos, &self.ropes[layer_id], Some(input_ids))?;
        }

        let h = self.hc_head_reduce(&h)?;

        let h = self.rmsnorm(&h, &self.norm_weight)?;

        let logits = self.head_forward(&h)?;

        Ok(logits)
    }

    fn embed_lookup(&self, input_ids: &[u32], bsz: usize) -> Result<GpuTensor> {
        let seqlen = input_ids.len();
        let dim = self.config.hidden_size;
        let device = self.device.clone();

        let row_indices: Vec<usize> = input_ids.iter().map(|&id| id as usize).collect();
        let embed_2d = GpuTensor {
            slice: self.embed.slice.clone(),
            shape: vec![self.config.vocab_size, dim],
            dtype: self.embed.dtype,
            device: device.clone(),
        };
        let gathered = embed_2d.gather_rows(&row_indices, dim)?;

        Ok(GpuTensor {
            slice: gathered.slice,
            shape: vec![bsz, seqlen, dim],
            dtype: DType::BF16,
            device,
        })
    }

    fn hc_expand(&self, x: &GpuTensor, hc: usize) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = x.shape[2];
        let device = x.device.clone();
        let total = bsz * seqlen;
        let elem_size = x.dtype.element_size();

        let out = GpuTensor::zeros(device.clone(), vec![bsz, seqlen, hc, dim], x.dtype)?;

        let stream = device.default_stream();
        {
            let (src_ptr, _src_guard) = x.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            let row_bytes = dim * elem_size;
            let hc_row_bytes = hc * row_bytes;

            for t in 0..total {
                let src_off = (t * row_bytes) as u64;
                let dst_off = (t * hc_row_bytes) as u64;
                unsafe {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + dst_off,
                        src_ptr + src_off,
                        row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
                for h in 1..hc {
                    unsafe {
                        cudarc::driver::sys::cuMemcpyAsync(
                            dst_ptr + dst_off + (h * row_bytes) as u64,
                            dst_ptr + dst_off,
                            row_bytes,
                            stream.cu_stream() as *mut _,
                        );
                    }
                }
            }
            stream.synchronize()?;
        }

        Ok(out)
    }

    fn hc_head_reduce(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = self.config.hidden_size;
        let hc = self.config.hc_mult;
        let device = x.device.clone();
        let total = bsz * seqlen;

        let x_flat_shape = vec![total, hc * dim];
        let x_flat = GpuTensor {
            slice: x.slice.clone(),
            shape: x_flat_shape,
            dtype: x.dtype,
            device: device.clone(),
        };

        let x_flat_f32 = self.cast_to_f32(&x_flat)?;
        let x_normed = self.rmsnorm_no_weight(&x_flat_f32)?;

        let mixes = self.gemm_f32(&x_normed, &self.hc_head_fn)?;

        let mixes_2d = GpuTensor {
            slice: mixes.slice.clone(),
            shape: vec![total, hc],
            dtype: DType::FP32,
            device: device.clone(),
        };
        let pre_bf16_out = GpuTensor::zeros(device.clone(), vec![total, hc], DType::BF16)?;

        let pre_bf16 = if self.kernels.call("hc_sigmoid_hc4", &[&mixes_2d, &self.hc_head_scale, &self.hc_head_base, &pre_bf16_out]).is_ok() {
            pre_bf16_out
        } else {
            let mixes_host = mixes.to_host()?;
            let mixes_f32: &[f32] = bytemuck::cast_slice(&mixes_host.data);
            let scale_host = self.hc_head_scale.to_host()?;
            let scale_f32: &[f32] = bytemuck::cast_slice(&scale_host.data);
            let base_host = self.hc_head_base.to_host()?;
            let base_f32: &[f32] = bytemuck::cast_slice(&base_host.data);
            let eps = self.config.hc_eps as f32;
            let mut pre_data = vec![0.0f32; total * hc];
            for t in 0..total {
                for j in 0..hc {
                    let mix_val = mixes_f32[t * hc + j];
                    let s = if !scale_f32.is_empty() { scale_f32[0] } else { 1.0 };
                    let b = if j < base_f32.len() { base_f32[j] } else { 0.0 };
                    pre_data[t * hc + j] = 1.0 / (1.0 + (-(mix_val * s + b)).exp()) + eps;
                }
            }
            let pre_cpu = CpuTensor::new(bytemuck::cast_slice(&pre_data).to_vec(), vec![total, hc], DType::FP32);
            let pre_gpu = GpuTensor::from_host(device.clone(), &pre_cpu)?;
            self.cast_to_bf16(&pre_gpu)?
        };

        let mut y = GpuTensor::zeros(device, vec![total, dim], DType::BF16)?;
        self.cublas.gemm_bf16_nn_strided_batched(
            1, dim, hc,
            &pre_bf16, x, &mut y,
            hc as i64, (hc * dim) as i64, dim as i64,
            total as i32,
            1.0, 0.0,
        )?;

        Ok(GpuTensor { slice: y.slice, shape: vec![bsz, seqlen, dim], dtype: DType::BF16, device: x.device.clone() })
    }

    fn cast_to_f32(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if x.dtype == DType::FP32 {
            return Ok(x.clone());
        }
        let n = x.shape.iter().product::<usize>();
        let last_dim = *x.shape.last().unwrap_or(&1);
        let device = x.device.clone();

        let kernel_name = match last_dim {
            4096 => Some("cast_bf16_to_f32_N4096"),
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

            if self.kernels.call(kname, &[&x_2d, &y_2d]).is_ok() {
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

            if self.kernels.call(kname, &[&x_2d, &y_2d]).is_ok() {
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

    fn rmsnorm(&self, x: &GpuTensor, weight: &GpuTensor) -> Result<GpuTensor> {
        let device = x.device.clone();
        let shape = x.shape.clone();
        let eps = self.config.rms_norm_eps as f32;
        let last_dim = shape.last().copied().unwrap_or(1);
        let n = shape.iter().product::<usize>();
        let m = n / last_dim;

        let kernel_name = match last_dim {
            4096 => Some("rmsnorm_N4096"),
            1024 => Some("rmsnorm_N1024"),
            512 => Some("rmsnorm_N512"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let w_f32 = self.cast_to_f32(&GpuTensor {
                slice: weight.slice.clone(),
                shape: vec![last_dim],
                dtype: weight.dtype,
                device: device.clone(),
            })?;
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::BF16)?;

            if self.kernels.call(kname, &[&x_2d, &w_f32, &y_2d]).is_ok() {
                return Ok(GpuTensor {
                    slice: y_2d.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let x_host = x.to_host()?;
        let w_host = weight.to_host()?;
        let x_bf16: &[half::bf16] = bytemuck::cast_slice(&x_host.data);
        let w_bf16: &[half::bf16] = bytemuck::cast_slice(&w_host.data);

        let out = crate::quant::rmsnorm_bf16(x_bf16, Some(w_bf16), last_dim, eps);
        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out).to_vec(), shape, DType::BF16);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn head_forward(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let device = x.device.clone();

        let last_only = bsz == 1 && seqlen > 1;
        let (m, x_slice) = if last_only {
            let row_indices = vec![seqlen - 1];
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![bsz * seqlen, dim],
                dtype: x.dtype,
                device: device.clone(),
            };
            let gathered = x_2d.gather_rows(&row_indices, dim)?;
            (1, gathered)
        } else {
            (bsz * seqlen, GpuTensor {
                slice: x.slice.clone(),
                shape: vec![bsz * seqlen, dim],
                dtype: x.dtype,
                device: device.clone(),
            })
        };

        let x_f32 = self.cast_to_f32(&x_slice)?;
        let w_f32 = self.cast_to_f32(&self.head_weight)?;

        let mut logits_f32 = GpuTensor::zeros(device.clone(), vec![m, vocab], DType::FP32)?;
        self.cublas.gemm_f32(m, vocab, dim, &x_f32, &w_f32, &mut logits_f32, 1.0, 0.0)?;

        let out_seqlen = if last_only { 1 } else { seqlen };
        let logits = GpuTensor {
            slice: logits_f32.slice,
            shape: vec![bsz, out_seqlen, vocab],
            dtype: logits_f32.dtype,
            device: logits_f32.device,
        };

        Ok(logits)
    }

    fn rmsnorm_no_weight(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let device = x.device.clone();
        let shape = x.shape.clone();
        let eps = self.config.rms_norm_eps as f32;
        let last_dim = shape.last().copied().unwrap_or(1);
        let n = shape.iter().product::<usize>();
        let m = n / last_dim;

        let kernel_name = match last_dim {
            1024 => Some("rmsnorm_no_weight_N1024"),
            _ => None,
        };

        if let Some(kname) = kernel_name {
            let x_2d = GpuTensor {
                slice: x.slice.clone(),
                shape: vec![m, last_dim],
                dtype: DType::BF16,
                device: device.clone(),
            };
            let y_2d = GpuTensor::zeros(device.clone(), vec![m, last_dim], DType::BF16)?;

            if self.kernels.call(kname, &[&x_2d, &y_2d]).is_ok() {
                return Ok(GpuTensor {
                    slice: y_2d.slice,
                    shape,
                    dtype: DType::BF16,
                    device,
                });
            }
        }

        let host = x.to_host()?;
        let data_f32: Vec<f32> = match host.dtype {
            DType::BF16 => {
                let bf16: &[half::bf16] = bytemuck::cast_slice(&host.data);
                bf16.iter().map(|v| v.to_f32()).collect()
            }
            DType::FP32 => bytemuck::cast_slice(&host.data).to_vec(),
            _ => return Err(anyhow!("unsupported dtype for rmsnorm")),
        };

        let out = crate::quant::rmsnorm_f32(&data_f32, last_dim, eps);
        let out_cpu = CpuTensor::new(bytemuck::cast_slice(&out).to_vec(), shape, DType::FP32);
        GpuTensor::from_host(device, &out_cpu)
    }

    fn gemm_f32(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[0];
        let device = a.device.clone();

        let mut c = GpuTensor::zeros(device, vec![m, n], DType::FP32)?;
        self.cublas.gemm_f32(m, n, k, a, b, &mut c, 1.0, 0.0)?;
        Ok(c)
    }
}
