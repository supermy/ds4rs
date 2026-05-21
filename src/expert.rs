use crate::cache::ThreeLevelCache;
use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::pinned::{PinnedBuffer, PinnedPool};
use crate::quant::{dequant_fp4_e2m1_to_bf16, dequant_fp8_e4m3_to_bf16};
use crate::tensor::{CpuTensor, GpuTensor};
use crate::weight::WeightLoader;
use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ExpertWeights {
    pub w1: GpuTensor,
    pub w1_scale: GpuTensor,
    pub w3: GpuTensor,
    pub w3_scale: GpuTensor,
    pub w2: GpuTensor,
    pub w2_scale: GpuTensor,
}

struct ExpertCpuWeights {
    w1: CpuTensor,
    w1_scale: CpuTensor,
    w3: CpuTensor,
    w3_scale: CpuTensor,
    w2: CpuTensor,
    w2_scale: CpuTensor,
}

pub struct ExpertScheduler {
    pub device: Arc<CudaContext>,
    pub config: Arc<ModelConfig>,
    pub expert_dtype: DType,
    pub inter_dim: usize,
    pub dim: usize,
    cpu_cache: HashMap<(usize, usize), ExpertCpuWeights>,
    three_level: ThreeLevelCache,
    pinned_pool: PinnedPool,
    transfer_stream: Option<Arc<cudarc::driver::CudaStream>>,
    prefetch_pending: HashMap<(usize, usize), Arc<ExpertWeights>>,
    prefetch_pinned: Vec<PinnedBuffer>,
    prefetch_depth: usize,
    max_prefetch_depth: usize,
    _vram_total_mb: usize,
    _vram_overhead_mb: usize,
    gpu_slots: GpuWeightSlots,
}

struct GpuWeightSlots {
    w1: Arc<cudarc::driver::CudaSlice<u8>>,
    w1_scale: Arc<cudarc::driver::CudaSlice<u8>>,
    w3: Arc<cudarc::driver::CudaSlice<u8>>,
    w3_scale: Arc<cudarc::driver::CudaSlice<u8>>,
    w2: Arc<cudarc::driver::CudaSlice<u8>>,
    w2_scale: Arc<cudarc::driver::CudaSlice<u8>>,
    w1_shape: Vec<usize>,
    w1_scale_shape: Vec<usize>,
    w3_shape: Vec<usize>,
    w3_scale_shape: Vec<usize>,
    w2_shape: Vec<usize>,
    w2_scale_shape: Vec<usize>,
}

impl ExpertScheduler {
    pub fn new(device: Arc<CudaContext>, config: Arc<ModelConfig>) -> Self {
        let expert_dtype = DType::from_config_str(&config.expert_dtype)
            .unwrap_or(DType::FP8E4M3);

        let expert_bytes = Self::expert_weight_bytes(&config);
        let gpu_slots = Self::compute_gpu_slots(&config, expert_bytes);
        let (ram_hot_mb, ram_cold_mb) = Self::compute_ram_config();

        let ssd_path = format!("{}/expert_cache", config.model_dir);
        let vram_total_mb = 16 * 1024usize;
        let vram_overhead_mb = 8 * 1024usize;
        let max_prefetch_depth = {
            let _expert_bytes_mb = expert_bytes / (1024 * 1024);
            let available_slots = gpu_slots / 4;
            (available_slots / config.n_routed_experts.max(1)).min(4).max(1)
        };
        let prefetch_depth = 1usize;

        let gpu_slots = {
            let stream = device.default_stream();
            let is_fp4 = expert_dtype == DType::FP4E2M1;
            let s_inter_dim = config.moe_intermediate_size;
            let s_dim = config.hidden_size;
            let w1_rows = s_inter_dim;
            let w1_cols_packed = if is_fp4 { s_dim / 2 } else { s_dim };
            let w1_scale_cols = if is_fp4 { s_dim / 32 } else { 0 };
            let w2_rows = s_dim;
            let w2_cols_packed = if is_fp4 { s_inter_dim / 2 } else { s_inter_dim };
            let w2_scale_cols = if is_fp4 { s_inter_dim / 32 } else { 0 };

            let w1_nbytes = w1_rows * w1_cols_packed;
            let w1_s_nbytes = w1_rows * w1_scale_cols;
            let w3_nbytes = w1_rows * w1_cols_packed;
            let w3_s_nbytes = w1_rows * w1_scale_cols;
            let w2_nbytes = w2_rows * w2_cols_packed;
            let w2_s_nbytes = w2_rows * w2_scale_cols;

            macro_rules! alloc_slot {
                ($nbytes:expr) => {
                    if $nbytes > 0 {
                        Arc::new(stream.alloc_zeros::<u8>($nbytes).expect("GPU slot alloc"))
                    } else {
                        Arc::new(stream.alloc_zeros::<u8>(1).expect("GPU slot alloc min"))
                    }
                };
            }

            GpuWeightSlots {
                w1: alloc_slot!(w1_nbytes),
                w1_scale: alloc_slot!(w1_s_nbytes),
                w3: alloc_slot!(w3_nbytes),
                w3_scale: alloc_slot!(w3_s_nbytes),
                w2: alloc_slot!(w2_nbytes),
                w2_scale: alloc_slot!(w2_s_nbytes),
                w1_shape: vec![w1_rows, w1_cols_packed],
                w1_scale_shape: vec![w1_rows, w1_scale_cols.max(1)],
                w3_shape: vec![w1_rows, w1_cols_packed],
                w3_scale_shape: vec![w1_rows, w1_scale_cols.max(1)],
                w2_shape: vec![w2_rows, w2_cols_packed],
                w2_scale_shape: vec![w2_rows, w2_scale_cols.max(1)],
            }
        };

        Self {
            device: device.clone(),
            inter_dim: config.moe_intermediate_size,
            dim: config.hidden_size,
            expert_dtype,
            config: config.clone(),
            cpu_cache: HashMap::new(),
            three_level: ThreeLevelCache::new(
                device.clone(),
                config,
                0,
                ram_hot_mb,
                ram_cold_mb,
                &ssd_path,
            ),
            pinned_pool: PinnedPool::new(expert_bytes),
            transfer_stream: device.new_stream().ok(),
            prefetch_pending: HashMap::new(),
            prefetch_pinned: Vec::new(),
            prefetch_depth,
            max_prefetch_depth: max_prefetch_depth.min(4),
            _vram_total_mb: vram_total_mb,
            _vram_overhead_mb: vram_overhead_mb,
            gpu_slots,
        }
    }

    fn expert_weight_bytes(config: &ModelConfig) -> usize {
        let dim = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let is_fp4 = config.expert_dtype == "fp4" || config.expert_dtype == "fp4e2m1";
        let bytes_per_elem = if is_fp4 { 1 } else { 1 };
        let w1 = dim * inter / (if is_fp4 { 2 } else { 1 }) * bytes_per_elem;
        let w3 = w1;
        let w2 = inter * dim / (if is_fp4 { 2 } else { 1 }) * bytes_per_elem;
        let scale_elems = if is_fp4 {
            2 * dim * inter / 32 + inter * dim / 32
        } else {
            0
        };
        w1 + w3 + w2 + scale_elems
    }

    fn compute_gpu_slots(_config: &ModelConfig, expert_bytes: usize) -> usize {
        let total_vram_mb: usize = 16 * 1024;
        let static_overhead_mb: usize = 11 * 1024;
        let kv_cache_mb: usize = 1 * 1024;
        let inference_overhead_mb: usize = 2 * 1024;
        let available_mb = total_vram_mb.saturating_sub(static_overhead_mb + kv_cache_mb + inference_overhead_mb);
        let expert_mb = (expert_bytes + 512 * 1024 - 1) / (1024 * 1024);
        if expert_mb == 0 { return 8; }
        let slots = available_mb / expert_mb.max(1);
        slots.max(8).min(64)
    }

    fn compute_ram_config() -> (usize, usize) {
        // 96GB 物理内存，按 90GB 使用，预留 6GB 给系统
        let total_ram_mb: usize = 90 * 1024;
        let system_overhead_mb: usize = 6 * 1024;
        let model_weights_mb: usize = 10 * 1024;
        let available_mb = total_ram_mb.saturating_sub(system_overhead_mb + model_weights_mb);
        let hot_mb = (available_mb as f64 * 0.3) as usize;
        let cold_mb = available_mb.saturating_sub(hot_mb);
        (hot_mb, cold_mb)
    }

    pub fn load_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        loader: &mut WeightLoader,
    ) -> Result<()> {
        let key = (layer_id, expert_id);
        if self.cpu_cache.contains_key(&key) {
            return Ok(());
        }

        let p = format!("layers.{}.ffn.experts.{}.", layer_id, expert_id);

        let w1 = loader.load(&(p.clone() + "w1.weight"))
            .with_context(|| format!("expert {}/{} w1", layer_id, expert_id))?;
        let w1 = self.fix_fp4_dtype(w1);
        let w1_scale = if loader.contains(&(p.clone() + "w1.scale")) {
            loader.load(&(p.clone() + "w1.scale"))?
        } else {
            CpuTensor::new(vec![], vec![0], DType::FP8E8M0)
        };
        let w3 = loader.load(&(p.clone() + "w3.weight"))
            .with_context(|| format!("expert {}/{} w3", layer_id, expert_id))?;
        let w3 = self.fix_fp4_dtype(w3);
        let w3_scale = if loader.contains(&(p.clone() + "w3.scale")) {
            loader.load(&(p.clone() + "w3.scale"))?
        } else {
            CpuTensor::new(vec![], vec![0], DType::FP8E8M0)
        };
        let w2 = loader.load(&(p.clone() + "w2.weight"))
            .with_context(|| format!("expert {}/{} w2", layer_id, expert_id))?;
        let w2 = self.fix_fp4_dtype(w2);
        let w2_scale = if loader.contains(&(p.clone() + "w2.scale")) {
            loader.load(&(p.clone() + "w2.scale"))?
        } else {
            CpuTensor::new(vec![], vec![0], DType::FP8E8M0)
        };

        self.cpu_cache.insert(key, ExpertCpuWeights {
            w1, w1_scale, w3, w3_scale, w2, w2_scale,
        });

        Ok(())
    }

    pub fn preload_experts(&mut self, layer_id: usize, expert_ids: &[usize], loader: &mut WeightLoader) -> Result<()> {
        for &eid in expert_ids {
            self.load_expert(layer_id, eid, loader)?;
        }
        Ok(())
    }

    pub fn get_expert_gpu(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        raw: bool,
    ) -> Result<Arc<ExpertWeights>> {
        let key = (layer_id, expert_id);

        if let Some(weights) = self.three_level.gpu.get(layer_id, expert_id) {
            if !raw && weights.w1.dtype == DType::FP4E2M1 {
                self.three_level.gpu.evict(layer_id, expert_id);
            } else {
                return Ok(weights);
            }
        }

        if let Some(weights) = self.prefetch_pending.remove(&key) {
            if let Some(ref stream) = self.transfer_stream {
                stream.synchronize().ok();
            }
            let _ = self.three_level.gpu.put(layer_id, expert_id, Arc::clone(&weights));
            return Ok(weights);
        }

        let cpu = match self.cpu_cache.get(&key) {
            Some(c) => c,
            None => {
                if self.load_from_ssd(layer_id, expert_id)? {
                    self.cpu_cache.get(&key)
                        .ok_or_else(|| anyhow!("expert {}/{} loaded but missing from CPU cache", layer_id, expert_id))?
                } else {
                    return Err(anyhow!("expert {}/{} not found in any cache", layer_id, expert_id));
                }
            }
        };

        let weights = if raw && self.expert_dtype == DType::FP4E2M1 {
            let mut b1 = self.pinned_pool.get(cpu.w1.nbytes())?;
            let w1 = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w1, &mut b1)?;
            let mut b2 = self.pinned_pool.get(cpu.w1_scale.nbytes())?;
            let w1_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w1_scale, &mut b2)?;
            let mut b3 = self.pinned_pool.get(cpu.w3.nbytes())?;
            let w3 = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w3, &mut b3)?;
            let mut b4 = self.pinned_pool.get(cpu.w3_scale.nbytes())?;
            let w3_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w3_scale, &mut b4)?;
            let mut b5 = self.pinned_pool.get(cpu.w2.nbytes())?;
            let w2 = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w2, &mut b5)?;
            let mut b6 = self.pinned_pool.get(cpu.w2_scale.nbytes())?;
            let w2_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w2_scale, &mut b6)?;
            Arc::new(ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale })
        } else {
            let w1_bf16 = self.dequant_to_bf16(&cpu.w1, &cpu.w1_scale)?;
            let w3_bf16 = self.dequant_to_bf16(&cpu.w3, &cpu.w3_scale)?;
            let w2_bf16 = self.dequant_to_bf16(&cpu.w2, &cpu.w2_scale)?;
            let mut b1 = self.pinned_pool.get(w1_bf16.nbytes())?;
            let w1 = GpuTensor::from_host_pinned(self.device.clone(), &w1_bf16, &mut b1)?;
            let mut b2 = self.pinned_pool.get(cpu.w1_scale.nbytes())?;
            let w1_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w1_scale, &mut b2)?;
            let mut b3 = self.pinned_pool.get(w3_bf16.nbytes())?;
            let w3 = GpuTensor::from_host_pinned(self.device.clone(), &w3_bf16, &mut b3)?;
            let mut b4 = self.pinned_pool.get(cpu.w3_scale.nbytes())?;
            let w3_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w3_scale, &mut b4)?;
            let mut b5 = self.pinned_pool.get(w2_bf16.nbytes())?;
            let w2 = GpuTensor::from_host_pinned(self.device.clone(), &w2_bf16, &mut b5)?;
            let mut b6 = self.pinned_pool.get(cpu.w2_scale.nbytes())?;
            let w2_scale = GpuTensor::from_host_pinned(self.device.clone(), &cpu.w2_scale, &mut b6)?;
            Arc::new(ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale })
        };

        let _ = self.three_level.gpu.put(layer_id, expert_id, Arc::clone(&weights));

        Ok(weights)
    }

    fn dequant_to_bf16(&self, weight: &CpuTensor, scale: &CpuTensor) -> Result<CpuTensor> {
        match weight.dtype {
            DType::BF16 => Ok(CpuTensor::new(weight.data.clone(), weight.shape.clone(), DType::BF16)),
            DType::FP8E4M3 => {
                dequant_fp8_e4m3_to_bf16(&weight.data, &scale.data, &weight.shape)
            }
            DType::FP4E2M1 => {
                let logical_k = self.inter_dim.max(self.dim);
                dequant_fp4_e2m1_to_bf16(&weight.data, &scale.data, &weight.shape, logical_k)
            }
            _ => Err(anyhow!("unsupported expert weight dtype: {}", weight.dtype)),
        }
    }

    /// 修正 safetensors 加载的 FP4 权重 dtype
    /// safetensors 中 FP4 权重以 I8 格式存储（2个FP4值打包为1个int8字节），
    /// weight.rs 的 safetensors_dtype_to_dtype 将 I8 映射为 UINT8，
    /// 但 FP4 GEMM 路径需要 DType::FP4E2M1 才能正确进入快速路径
    fn fix_fp4_dtype(&self, tensor: CpuTensor) -> CpuTensor {
        if self.expert_dtype == DType::FP4E2M1 && tensor.dtype == DType::UINT8 {
            CpuTensor::new(tensor.data, tensor.shape, DType::FP4E2M1)
        } else {
            tensor
        }
    }

    pub fn expert_count(&self) -> usize {
        self.cpu_cache.len()
    }

    pub fn cache_stats(&self) -> String {
        self.three_level.cache_stats().to_string()
    }

    pub fn prefetch_to_ssd(&mut self, layer_id: usize, expert_id: usize, loader: &mut WeightLoader) -> Result<()> {
        self.load_expert(layer_id, expert_id, loader)?;
        let key = (layer_id, expert_id);
        if let Some(cpu) = self.cpu_cache.get(&key) {
            let serialized = self.serialize_expert(cpu)?;
            self.three_level.ssd.put_indexed(layer_id, expert_id, &serialized)?;
        }
        Ok(())
    }

    fn serialize_expert(&self, expert: &ExpertCpuWeights) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        Self::write_tensor(&mut buf, &expert.w1)?;
        Self::write_tensor(&mut buf, &expert.w1_scale)?;
        Self::write_tensor(&mut buf, &expert.w3)?;
        Self::write_tensor(&mut buf, &expert.w3_scale)?;
        Self::write_tensor(&mut buf, &expert.w2)?;
        Self::write_tensor(&mut buf, &expert.w2_scale)?;
        Ok(buf)
    }

    fn write_tensor(buf: &mut Vec<u8>, tensor: &CpuTensor) -> Result<()> {
        let ndim = tensor.shape.len() as u32;
        buf.extend_from_slice(&ndim.to_le_bytes());
        for &dim in &tensor.shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        let dtype_tag = match tensor.dtype {
            DType::BF16 => 0u32,
            DType::FP8E4M3 => 1u32,
            DType::FP8E8M0 => 2u32,
            DType::FP4E2M1 => 3u32,
            DType::FP32 => 4u32,
            DType::INT32 => 5u32,
            _ => 99u32,
        };
        buf.extend_from_slice(&dtype_tag.to_le_bytes());
        let len = tensor.data.len() as u64;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&tensor.data);
        Ok(())
    }

    fn deserialize_expert(&self, data: &[u8]) -> Result<ExpertCpuWeights> {
        let mut offset = 0;
        let w1 = Self::read_tensor(data, &mut offset)?;
        let w1_scale = Self::read_tensor(data, &mut offset)?;
        let w3 = Self::read_tensor(data, &mut offset)?;
        let w3_scale = Self::read_tensor(data, &mut offset)?;
        let w2 = Self::read_tensor(data, &mut offset)?;
        let w2_scale = Self::read_tensor(data, &mut offset)?;
        Ok(ExpertCpuWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale })
    }

    fn read_tensor(data: &[u8], offset: &mut usize) -> Result<CpuTensor> {
        let ndim = u32::from_le_bytes(data[*offset..*offset+4].try_into()?) as usize;
        *offset += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes(data[*offset..*offset+8].try_into()?) as usize;
            shape.push(dim);
            *offset += 8;
        }
        let dtype_tag = u32::from_le_bytes(data[*offset..*offset+4].try_into()?);
        *offset += 4;
        let dtype = match dtype_tag {
            0 => DType::BF16,
            1 => DType::FP8E4M3,
            2 => DType::FP8E8M0,
            3 => DType::FP4E2M1,
            4 => DType::FP32,
            5 => DType::INT32,
            _ => return Err(anyhow!("unknown dtype tag {}", dtype_tag)),
        };
        let len = u64::from_le_bytes(data[*offset..*offset+8].try_into()?) as usize;
        *offset += 8;
        let tensor_data = data[*offset..*offset+len].to_vec();
        *offset += len;
        Ok(CpuTensor::new(tensor_data, shape, dtype))
    }

    pub fn load_from_ssd(&mut self, layer_id: usize, expert_id: usize) -> Result<bool> {
        let key = (layer_id, expert_id);
        if self.cpu_cache.contains_key(&key) {
            return Ok(true);
        }

        if let Some(data) = self.three_level.ram.get_copy(layer_id, expert_id) {
            if let Ok(expert) = self.deserialize_expert(&data) {
                self.cpu_cache.insert(key, expert);
                return Ok(true);
            }
        }

        if let Some(mmap) = self.three_level.ssd.mmap_prefetch(layer_id, expert_id) {
            let expert = self.deserialize_expert(&mmap)?;
            self.three_level.ram.put(layer_id, expert_id, mmap.to_vec());
            self.cpu_cache.insert(key, expert);
            return Ok(true);
        }

        if let Some(data) = self.three_level.ssd.get(layer_id, expert_id) {
            let expert = self.deserialize_expert(&data)?;
            self.three_level.ram.put(layer_id, expert_id, data);
            self.cpu_cache.insert(key, expert);
            return Ok(true);
        }

        Ok(false)
    }

    pub fn ensure_expert_loaded(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        loader: &mut WeightLoader,
    ) -> Result<()> {
        if self.cpu_cache.contains_key(&(layer_id, expert_id)) {
            return Ok(());
        }
        if self.load_from_ssd(layer_id, expert_id)? {
            return Ok(());
        }
        self.load_expert(layer_id, expert_id, loader)
    }

    pub fn prefetch_next_layer(
        &mut self,
        current_layer: usize,
        expert_ids: &[usize],
        loader: &mut WeightLoader,
    ) -> Result<()> {
        let next_layer = current_layer + 1;
        if next_layer >= self.config.num_hidden_layers {
            return Ok(());
        }

        for &eid in expert_ids {
            let key = (next_layer, eid);
            if self.cpu_cache.contains_key(&key) {
                continue;
            }
            if self.three_level.gpu.contains(next_layer, eid)
                || self.prefetch_pending.contains_key(&key)
            {
                continue;
            }

            if self.three_level.ssd.contains(next_layer, eid) {
                if let Some(data) = self.three_level.ssd.get(next_layer, eid) {
                    if let Ok(expert) = self.deserialize_expert(&data) {
                        self.three_level.ram.put(next_layer, eid, data);
                        self.cpu_cache.insert(key, expert);
                    }
                }
            } else {
                let _ = self.load_expert(next_layer, eid, loader);
            }
        }

        Ok(())
    }

    pub fn finalize_prefetch(&mut self) {
        if let Some(ref stream) = self.transfer_stream {
            stream.synchronize().ok();
        }
        self.prefetch_pinned.clear();
        let pending: Vec<_> = self.prefetch_pending.drain().collect();
        for (key, weights) in pending {
            let _ = self.three_level.gpu.put(key.0, key.1, weights);
        }
    }

    pub fn adapt(&mut self) {
        self.three_level.adapt();
        let gpu_hit = self.three_level.gpu.hit_rate();
        if gpu_hit > 0.85 {
            self.prefetch_depth = 1;
        } else if gpu_hit < 0.4 {
            self.prefetch_depth = self.max_prefetch_depth;
        } else {
            self.prefetch_depth = (self.max_prefetch_depth + 1) / 2;
        }
    }

    pub fn prefetch_layers_ahead(
        &mut self,
        current_layer: usize,
        expert_ids: &[usize],
        loader: &mut WeightLoader,
    ) -> Result<()> {
        let depth = self.prefetch_depth;
        for offset in 1..=depth {
            let target_layer = current_layer + offset;
            if target_layer >= self.config.num_hidden_layers {
                break;
            }

            for &eid in expert_ids {
                if !self.cpu_cache.contains_key(&(target_layer, eid))
                    && !self.three_level.ram.contains(target_layer, eid)
                {
                    if self.three_level.ssd.contains(target_layer, eid) {
                        if let Some(data) = self.three_level.ssd.get(target_layer, eid) {
                            if let Ok(expert) = self.deserialize_expert(&data) {
                                self.three_level.ram.put(target_layer, eid, data);
                                self.cpu_cache.insert((target_layer, eid), expert);
                            }
                        }
                    } else {
                        let _ = self.load_expert(target_layer, eid, loader);
                    }
                }
            }

            self.prefetch_next_layer(target_layer.saturating_sub(1), expert_ids, loader)?;
        }
        Ok(())
    }

    pub fn vram_utilization(&self) -> f64 {
        let used = self.three_level.gpu.len();
        let total = self.three_level.gpu.max_capacity();
        if total == 0 { return 0.0; }
        used as f64 / total as f64
    }

    pub fn clear_gpu_cache(&mut self) {
        self.three_level.gpu.clear();
    }

    pub fn upload_expert_raw(&mut self, layer_id: usize, expert_id: usize) -> Result<ExpertWeights> {
        let key = (layer_id, expert_id);

        let cpu = match self.cpu_cache.get(&key) {
            Some(c) => c,
            None => {
                if self.load_from_ssd(layer_id, expert_id)? {
                    self.cpu_cache.get(&key)
                        .ok_or_else(|| anyhow!("expert {}/{} loaded but missing from CPU cache", layer_id, expert_id))?
                } else {
                    return Err(anyhow!("expert {}/{} not found in any cache", layer_id, expert_id));
                }
            }
        };

        let slots = &self.gpu_slots;
        let device = &self.device;

        let copy_to_slot = |slot: &Arc<cudarc::driver::CudaSlice<u8>>, cpu_tensor: &CpuTensor, shape: Vec<usize>, dtype: DType| -> Result<GpuTensor> {
            let nbytes = cpu_tensor.nbytes();
            let tmp = GpuTensor {
                slice: Arc::clone(slot),
                shape: shape.clone(),
                dtype,
                device: device.clone(),
            };
            let dst_ptr = tmp.device_ptr();
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoD_v2(
                    dst_ptr as cudarc::driver::sys::CUdeviceptr,
                    cpu_tensor.data.as_ptr() as *const std::ffi::c_void,
                    nbytes,
                );
            }
            Ok(tmp)
        };

        if self.expert_dtype == DType::FP4E2M1 {
            let w1 = copy_to_slot(&slots.w1, &cpu.w1, slots.w1_shape.clone(), DType::FP4E2M1)?;
            let w1_scale = copy_to_slot(&slots.w1_scale, &cpu.w1_scale, slots.w1_scale_shape.clone(), DType::FP8E8M0)?;
            let w3 = copy_to_slot(&slots.w3, &cpu.w3, slots.w3_shape.clone(), DType::FP4E2M1)?;
            let w3_scale = copy_to_slot(&slots.w3_scale, &cpu.w3_scale, slots.w3_scale_shape.clone(), DType::FP8E8M0)?;
            let w2 = copy_to_slot(&slots.w2, &cpu.w2, slots.w2_shape.clone(), DType::FP4E2M1)?;
            let w2_scale = copy_to_slot(&slots.w2_scale, &cpu.w2_scale, slots.w2_scale_shape.clone(), DType::FP8E8M0)?;
            Ok(ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale })
        } else {
            let w1_bf16 = self.dequant_to_bf16(&cpu.w1, &cpu.w1_scale)?;
            let w3_bf16 = self.dequant_to_bf16(&cpu.w3, &cpu.w3_scale)?;
            let w2_bf16 = self.dequant_to_bf16(&cpu.w2, &cpu.w2_scale)?;
            let w1 = copy_to_slot(&slots.w1, &w1_bf16, slots.w1_shape.clone(), DType::BF16)?;
            let w1_scale = copy_to_slot(&slots.w1_scale, &cpu.w1_scale, slots.w1_scale_shape.clone(), DType::FP8E8M0)?;
            let w3 = copy_to_slot(&slots.w3, &w3_bf16, slots.w3_shape.clone(), DType::BF16)?;
            let w3_scale = copy_to_slot(&slots.w3_scale, &cpu.w3_scale, slots.w3_scale_shape.clone(), DType::FP8E8M0)?;
            let w2 = copy_to_slot(&slots.w2, &w2_bf16, slots.w2_shape.clone(), DType::BF16)?;
            let w2_scale = copy_to_slot(&slots.w2_scale, &cpu.w2_scale, slots.w2_scale_shape.clone(), DType::FP8E8M0)?;
            Ok(ExpertWeights { w1, w1_scale, w3, w3_scale, w2, w2_scale })
        }
    }
}
