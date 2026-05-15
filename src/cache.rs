use crate::config::ModelConfig;
use crate::expert::ExpertWeights;
use crate::tensor::GpuTensor;
use anyhow::{anyhow, Result};
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::sync::Arc;

pub struct GpuExpertCache {
    #[allow(dead_code)]
    device: Arc<CudaContext>,
    #[allow(dead_code)]
    config: Arc<ModelConfig>,
    slots: HashMap<(usize, usize), GpuExpertSlot>,
    freq: HashMap<(usize, usize), u64>,
    max_slots: usize,
    total_access: u64,
}

struct GpuExpertSlot {
    w1: GpuTensor,
    w1_scale: GpuTensor,
    w3: GpuTensor,
    w3_scale: GpuTensor,
    w2: GpuTensor,
    w2_scale: GpuTensor,
}

impl GpuExpertCache {
    pub fn new(device: Arc<CudaContext>, config: Arc<ModelConfig>, max_slots: usize) -> Self {
        Self {
            device,
            config,
            slots: HashMap::new(),
            freq: HashMap::new(),
            max_slots,
            total_access: 0,
        }
    }

    pub fn get(&mut self, layer_id: usize, expert_id: usize) -> Option<ExpertWeights> {
        let key = (layer_id, expert_id);
        if self.slots.contains_key(&key) {
            *self.freq.entry(key).or_insert(0) += 1;
            self.total_access += 1;
            let slot = self.slots.get(&key)?;
            Some(ExpertWeights {
                w1: slot.w1.clone(),
                w1_scale: slot.w1_scale.clone(),
                w3: slot.w3.clone(),
                w3_scale: slot.w3_scale.clone(),
                w2: slot.w2.clone(),
                w2_scale: slot.w2_scale.clone(),
            })
        } else {
            None
        }
    }

    pub fn put(&mut self, layer_id: usize, expert_id: usize, weights: ExpertWeights) -> Result<()> {
        let key = (layer_id, expert_id);

        if self.slots.len() >= self.max_slots && !self.slots.contains_key(&key) {
            self.evict_lfu()?;
        }

        self.slots.insert(key, GpuExpertSlot {
            w1: weights.w1,
            w1_scale: weights.w1_scale,
            w3: weights.w3,
            w3_scale: weights.w3_scale,
            w2: weights.w2,
            w2_scale: weights.w2_scale,
        });
        *self.freq.entry(key).or_insert(1) += 1;

        Ok(())
    }

    pub fn contains(&self, layer_id: usize, expert_id: usize) -> bool {
        self.slots.contains_key(&(layer_id, expert_id))
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn max_capacity(&self) -> usize {
        self.max_slots
    }

    fn evict_lfu(&mut self) -> Result<()> {
        if self.slots.is_empty() {
            return Err(anyhow!("no slots to evict"));
        }

        let mut min_freq = u64::MAX;
        let mut evict_key = (0usize, 0usize);

        for (&key, &freq) in &self.freq {
            if freq < min_freq && self.slots.contains_key(&key) {
                min_freq = freq;
                evict_key = key;
            }
        }

        self.slots.remove(&evict_key);
        self.freq.remove(&evict_key);
        Ok(())
    }

    pub fn prefetch_slots(&self) -> usize {
        self.max_slots.saturating_sub(self.slots.len())
    }
}

pub struct RamExpertCache {
    hot: HashMap<(usize, usize), Vec<u8>>,
    cold: HashMap<(usize, usize), Vec<u8>>,
    hot_capacity: usize,
    cold_capacity: usize,
    hot_size: usize,
    cold_size: usize,
}

impl RamExpertCache {
    pub fn new(hot_capacity_mb: usize, cold_capacity_mb: usize) -> Self {
        Self {
            hot: HashMap::new(),
            cold: HashMap::new(),
            hot_capacity: hot_capacity_mb * 1024 * 1024,
            cold_capacity: cold_capacity_mb * 1024 * 1024,
            hot_size: 0,
            cold_size: 0,
        }
    }

    pub fn get(&mut self, layer_id: usize, expert_id: usize) -> Option<Vec<u8>> {
        let key = (layer_id, expert_id);
        if self.hot.contains_key(&key) {
            return self.hot.get(&key).cloned();
        }
        if let Some(data) = self.cold.remove(&key) {
            let size = data.len();
            self.cold_size -= size;
            while self.hot_size + size > self.hot_capacity && !self.hot.is_empty() {
                self.demote_hot();
            }
            self.hot.insert(key, data);
            self.hot_size += size;
            return self.hot.get(&key).cloned();
        }
        None
    }

    pub fn put(&mut self, layer_id: usize, expert_id: usize, data: Vec<u8>) {
        let key = (layer_id, expert_id);
        let size = data.len();

        if self.hot.contains_key(&key) || self.cold.contains_key(&key) {
            return;
        }

        if self.hot_size + size <= self.hot_capacity {
            self.hot.insert(key, data);
            self.hot_size += size;
        } else if self.cold_size + size <= self.cold_capacity {
            self.cold.insert(key, data);
            self.cold_size += size;
        }
    }

    pub fn contains(&self, layer_id: usize, expert_id: usize) -> bool {
        self.hot.contains_key(&(layer_id, expert_id))
            || self.cold.contains_key(&(layer_id, expert_id))
    }

    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    pub fn cold_len(&self) -> usize {
        self.cold.len()
    }

    fn demote_hot(&mut self) {
        if let Some((key, data)) = self.hot.iter().next().map(|(k, v)| (*k, v.clone())) {
            let size = data.len();
            self.hot.remove(&key);
            self.hot_size -= size;
            if self.cold_size + size <= self.cold_capacity {
                self.cold.insert(key, data);
                self.cold_size += size;
            }
        }
    }
}

pub struct SsdExpertCache {
    base_path: String,
    index: HashMap<(usize, usize), u64>,
}

impl SsdExpertCache {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            index: HashMap::new(),
        }
    }

    pub fn get(&self, layer_id: usize, expert_id: usize) -> Option<Vec<u8>> {
        let key = (layer_id, expert_id);
        if !self.index.contains_key(&key) {
            return None;
        }

        let path = self.expert_path(layer_id, expert_id);
        std::fs::read(path).ok()
    }

    pub fn put(&self, layer_id: usize, expert_id: usize, data: &[u8]) -> Result<()> {
        let path = self.expert_path(layer_id, expert_id);
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, data)?;
        Ok(())
    }

    pub fn put_indexed(&mut self, layer_id: usize, expert_id: usize, data: &[u8]) -> Result<()> {
        self.put(layer_id, expert_id, data)?;
        self.index.insert((layer_id, expert_id), 0);
        Ok(())
    }

    pub fn mmap_prefetch(&self, layer_id: usize, expert_id: usize) -> Option<memmap2::Mmap> {
        let path = self.expert_path(layer_id, expert_id);
        if !std::path::Path::new(&path).exists() {
            return None;
        }
        let file = std::fs::File::open(&path).ok()?;
        unsafe { memmap2::Mmap::map(&file).ok() }
    }

    pub fn contains(&self, layer_id: usize, expert_id: usize) -> bool {
        self.index.contains_key(&(layer_id, expert_id))
    }

    pub fn scan_index(&mut self) -> usize {
        self.index.clear();
        let experts_dir = std::path::Path::new(&self.base_path).join("experts");
        if !experts_dir.exists() {
            return 0;
        }

        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(&experts_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some((layer_str, expert_str)) = name.split_once('_') {
                    if let (Ok(layer_id), Ok(expert_id)) = (layer_str.parse::<usize>(), expert_str.parse::<usize>()) {
                        self.index.insert((layer_id, expert_id), 0);
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn expert_path(&self, layer_id: usize, expert_id: usize) -> String {
        format!("{}/experts/{}_{}", self.base_path, layer_id, expert_id)
    }
}

pub struct ThreeLevelCache {
    pub gpu: GpuExpertCache,
    pub ram: RamExpertCache,
    pub ssd: SsdExpertCache,
}

impl ThreeLevelCache {
    pub fn new(
        device: Arc<CudaContext>,
        config: Arc<ModelConfig>,
        gpu_slots: usize,
        ram_hot_mb: usize,
        ram_cold_mb: usize,
        ssd_path: &str,
    ) -> Self {
        let mut ssd = SsdExpertCache::new(ssd_path);
        ssd.scan_index();
        Self {
            gpu: GpuExpertCache::new(device, config, gpu_slots),
            ram: RamExpertCache::new(ram_hot_mb, ram_cold_mb),
            ssd,
        }
    }

    pub fn get_expert(&mut self, layer_id: usize, expert_id: usize) -> Option<ExpertWeights> {
        if let Some(weights) = self.gpu.get(layer_id, expert_id) {
            return Some(weights);
        }
        None
    }

    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            gpu_slots_used: self.gpu.len(),
            gpu_slots_total: self.gpu.max_capacity(),
            ram_hot_entries: self.ram.hot_len(),
            ram_cold_entries: self.ram.cold_len(),
            ssd_entries: self.ssd.index.len(),
        }
    }
}

pub struct CacheStats {
    pub gpu_slots_used: usize,
    pub gpu_slots_total: usize,
    pub ram_hot_entries: usize,
    pub ram_cold_entries: usize,
    pub ssd_entries: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU[{}/{}] RAM[hot:{} cold:{}] SSD[{}]",
            self.gpu_slots_used, self.gpu_slots_total,
            self.ram_hot_entries, self.ram_cold_entries,
            self.ssd_entries
        )
    }
}
