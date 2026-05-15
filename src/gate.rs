use crate::config::ModelConfig;
use crate::cublas::CublasHandle;
use crate::dtype::DType;
use crate::tensor::{CpuTensor, GpuTensor};
use crate::tvm_ffi::KernelRegistry;
use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;

pub struct Gate {
    pub weight: GpuTensor,
    pub score_func: ScoreFunc,
    pub topk: usize,
    pub route_scale: f32,
    pub n_routed_experts: usize,
    pub cublas: Arc<CublasHandle>,
    pub kernels: Arc<KernelRegistry>,
    bias_cpu: Option<Vec<f32>>,
    bias_gpu: Option<GpuTensor>,
    tid2eid_cpu: Option<Vec<i32>>,
    tid2eid_cols: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreFunc {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

impl ScoreFunc {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "softmax" => ScoreFunc::Softmax,
            "sigmoid" => ScoreFunc::Sigmoid,
            _ => ScoreFunc::SqrtSoftplus,
        }
    }
}

pub struct GateOutput {
    pub weights: GpuTensor,
    pub indices: GpuTensor,
}

impl Gate {
    pub fn new(
        config: &ModelConfig,
        _layer_id: usize,
        device: Arc<CudaContext>,
        cublas: Arc<CublasHandle>,
        kernels: Arc<KernelRegistry>,
        weight: GpuTensor,
        bias: Option<GpuTensor>,
        tid2eid: Option<GpuTensor>,
    ) -> Self {
        let score_func = ScoreFunc::from_str(&config.scoring_func);

        let (bias_cpu, bias_gpu) = bias.map(|b| {
            let host = b.to_host().unwrap();
            let bf16: &[half::bf16] = bytemuck::cast_slice(&host.data);
            let cpu: Vec<f32> = bf16.iter().map(|v| v.to_f32()).collect();
            let n = cpu.len();
            let cpu_tensor = CpuTensor::new(
                bytemuck::cast_slice(&cpu).to_vec(),
                vec![n],
                DType::FP32,
            );
            let gpu = GpuTensor::from_host(device.clone(), &cpu_tensor).ok();
            (Some(cpu), gpu)
        }).unwrap_or((None, None));

        let (tid2eid_cpu, tid2eid_cols) = tid2eid.map(|t| {
            let host = t.to_host().unwrap();
            let cols = host.shape.get(1).copied().unwrap_or(1);
            let i32_data: &[i32] = bytemuck::cast_slice(&host.data);
            (i32_data.to_vec(), cols)
        }).map(|(d, c)| (Some(d), Some(c))).unwrap_or((None, None));

        Self {
            weight,
            score_func,
            topk: config.num_experts_per_tok,
            route_scale: config.routed_scaling_factor,
            n_routed_experts: config.n_routed_experts,
            cublas,
            kernels,
            bias_cpu,
            bias_gpu,
            tid2eid_cpu,
            tid2eid_cols,
        }
    }

    pub fn forward(
        &self,
        x: &GpuTensor,
        input_ids: Option<&[u32]>,
    ) -> Result<GateOutput> {
        let bsz = x.shape[0];
        let seqlen = x.shape[1];
        let dim = x.shape[2];
        let device = x.device.clone();

        let x_flat = GpuTensor {
            slice: x.slice.clone(),
            shape: vec![bsz * seqlen, dim],
            dtype: x.dtype,
            device: device.clone(),
        };

        let x_f32 = self.cast_to_f32(&x_flat)?;
        let w_f32 = self.cast_to_f32(&self.weight)?;

        let m = x_f32.shape[0];
        let k = x_f32.shape[1];
        let n = w_f32.shape[0];

        let mut scores = GpuTensor::zeros(device.clone(), vec![m, n], DType::FP32)?;
        self.cublas.gemm_f32(m, n, k, &x_f32, &w_f32, &mut scores, 1.0, 0.0)?;

        let use_hash = self.tid2eid_cpu.is_some() && input_ids.is_some();

        if !use_hash {
            if let Ok(output) = self.try_route_gpu(&scores, bsz, seqlen) {
                return Ok(output);
            }
        }

        let scores_host = scores.to_host()?;
        let scores_f32: &[f32] = bytemuck::cast_slice(&scores_host.data);

        let (weights, indices) = self.route_scores_cpu(
            scores_f32,
            self.bias_cpu.as_deref(),
            self.tid2eid_cpu.as_deref(),
            self.tid2eid_cols,
            input_ids,
            bsz,
            seqlen,
        )?;

        let weights_cpu = CpuTensor::new(
            bytemuck::cast_slice(&weights).to_vec(),
            vec![bsz, seqlen, self.topk],
            DType::FP32,
        );
        let indices_cpu = CpuTensor::new(
            bytemuck::cast_slice(&indices).to_vec(),
            vec![bsz, seqlen, self.topk],
            DType::INT32,
        );

        let weights_gpu = GpuTensor::from_host(device.clone(), &weights_cpu)?;
        let indices_gpu = GpuTensor::from_host(device, &indices_cpu)?;

        Ok(GateOutput {
            weights: weights_gpu,
            indices: indices_gpu,
        })
    }

    fn try_route_gpu(
        &self,
        scores: &GpuTensor,
        bsz: usize,
        seqlen: usize,
    ) -> Result<GateOutput> {
        let m = bsz * seqlen;
        let device = scores.device.clone();

        let kernel_name = self.gpu_route_kernel_name();

        let scores_2d = GpuTensor {
            slice: scores.slice.clone(),
            shape: vec![m, self.n_routed_experts],
            dtype: DType::FP32,
            device: device.clone(),
        };

        let topk_weights = GpuTensor::zeros(
            device.clone(),
            vec![m, self.topk],
            DType::FP32,
        )?;
        let topk_indices = GpuTensor::zeros(
            device.clone(),
            vec![m, self.topk],
            DType::INT32,
        )?;

        if let Some(bias_gpu) = self.bias_gpu.as_ref() {
            self.kernels.call(
                &kernel_name,
                &[&scores_2d, bias_gpu, &topk_weights, &topk_indices],
            )?;
        } else {
            let n = self.n_routed_experts;
            let zero_bias = GpuTensor::zeros(device.clone(), vec![n], DType::FP32)?;
            self.kernels.call(
                &kernel_name,
                &[&scores_2d, &zero_bias, &topk_weights, &topk_indices],
            )?;
        }

        Ok(GateOutput {
            weights: GpuTensor {
                slice: topk_weights.slice,
                shape: vec![bsz, seqlen, self.topk],
                dtype: DType::FP32,
                device: device.clone(),
            },
            indices: GpuTensor {
                slice: topk_indices.slice,
                shape: vec![bsz, seqlen, self.topk],
                dtype: DType::INT32,
                device,
            },
        })
    }

    fn gpu_route_kernel_name(&self) -> String {
        let func = match self.score_func {
            ScoreFunc::Sigmoid => "sigmoid",
            ScoreFunc::SqrtSoftplus => "sqrtsp",
            ScoreFunc::Softmax => "softmax",
        };
        format!(
            "moe_route_{}_N{}_topk{}",
            func, self.n_routed_experts, self.topk
        )
    }

    pub fn route_scores_cpu(
        &self,
        scores: &[f32],
        bias: Option<&[f32]>,
        tid2eid: Option<&[i32]>,
        tid2eid_cols: Option<usize>,
        input_ids: Option<&[u32]>,
        bsz: usize,
        seqlen: usize,
    ) -> Result<(Vec<f32>, Vec<i32>)> {
        let n = self.n_routed_experts;
        let total = bsz * seqlen;

        let mut activated = vec![0.0f32; total * n];
        for (i, &s) in scores.iter().enumerate() {
            activated[i] = match self.score_func {
                ScoreFunc::Softmax => s,
                ScoreFunc::Sigmoid => 1.0 / (1.0 + (-s).exp()),
                ScoreFunc::SqrtSoftplus => {
                    if s > 20.0 {
                        s.sqrt()
                    } else {
                        (1.0 + s.exp()).ln().sqrt()
                    }
                }
            };
        }

        if self.score_func == ScoreFunc::Softmax {
            for t in 0..total {
                let row = &mut activated[t * n..(t + 1) * n];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
                for v in row.iter_mut() {
                    *v = (*v - max).exp() / sum;
                }
            }
        }

        let original_scores = activated.clone();

        let mut topk_weights = vec![0.0f32; total * self.topk];
        let mut topk_indices = vec![0i32; total * self.topk];

        let use_hash = tid2eid.is_some() && input_ids.is_some();

        for t in 0..total {
            if use_hash {
                let ids = input_ids.unwrap();
                let token_id = ids[t % ids.len()] as usize;
                let cols = tid2eid_cols.unwrap();
                let t2e = tid2eid.unwrap();
                for k in 0..self.topk {
                    let expert_idx = if token_id * cols + k < t2e.len() {
                        t2e[token_id * cols + k] as usize
                    } else {
                        0
                    };
                    topk_indices[t * self.topk + k] = expert_idx as i32;
                    topk_weights[t * self.topk + k] = if expert_idx < n {
                        original_scores[t * n + expert_idx]
                    } else {
                        1.0 / self.topk as f32
                    };
                }
            } else {
                let scores_with_bias = if let Some(bias) = bias {
                    let mut s = vec![0.0f32; n];
                    for j in 0..n {
                        s[j] = activated[t * n + j] + bias[j];
                    }
                    s
                } else {
                    activated[t * n..(t + 1) * n].to_vec()
                };

                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| {
                    scores_with_bias[b].partial_cmp(&scores_with_bias[a]).unwrap_or(std::cmp::Ordering::Equal)
                });

                for k in 0..self.topk {
                    let expert_idx = idx[k];
                    topk_indices[t * self.topk + k] = expert_idx as i32;
                    topk_weights[t * self.topk + k] = original_scores[t * n + expert_idx];
                }
            }

            if self.score_func != ScoreFunc::Softmax {
                let sum: f32 = topk_weights[t * self.topk..(t + 1) * self.topk].iter().sum();
                if sum > 0.0 {
                    for w in topk_weights[t * self.topk..(t + 1) * self.topk].iter_mut() {
                        *w /= sum;
                    }
                }
            }

            for w in topk_weights[t * self.topk..(t + 1) * self.topk].iter_mut() {
                *w *= self.route_scale;
            }
        }

        Ok((topk_weights, topk_indices))
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
            _ => return Err(anyhow::anyhow!("cast_to_f32: unsupported dtype {:?}", host.dtype)),
        };
        GpuTensor::from_host(x.device.clone(), &out)
    }
}
