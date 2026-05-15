use crate::config::ModelConfig;
use crate::expert::ExpertWeights;
use crate::tensor::GpuTensor;
use anyhow::{anyhow, Result};
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::sync::Arc;

pub struct CacheHitStats {
    hits: u64,
    misses: u64,
}

impl CacheHitStats {
    pub fn new() -> Self {
        Self { hits: 0, misses: 0 }
    }

    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { return 1.0; }
        self.hits as f64 / total as f64
    }

    pub fn total(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

pub struct GpuExpertCache {
    #[allow(dead_code)]
    device: Arc<CudaContext>,
    #[allow(dead_code)]
    config: Arc<ModelConfig>,
    slots: HashMap<(usize, usize), GpuExpertSlot>,
    freq: HashMap<(usize, usize), u64>,
    last_access: HashMap<(usize, usize), u64>,
    max_slots: usize,
    min_slots: usize,
    access_counter: u64,
    stats: CacheHitStats,
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
        let min_slots = (max_slots / 4).max(4);
        Self {
            device,
            config,
            slots: HashMap::new(),
            freq: HashMap::new(),
            last_access: HashMap::new(),
            max_slots,
            min_slots,
            access_counter: 0,
            stats: CacheHitStats::new(),
        }
    }

    pub fn get(&mut self, layer_id: usize, expert_id: usize) -> Option<ExpertWeights> {
        let key = (layer_id, expert_id);
        if self.slots.contains_key(&key) {
            *self.freq.entry(key).or_insert(0) += 1;
            self.access_counter += 1;
            *self.last_access.entry(key).or_insert(0) = self.access_counter;
            self.stats.record_hit();
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
            self.stats.record_miss();
            None
        }
    }

    pub fn put(&mut self, layer_id: usize, expert_id: usize, weights: ExpertWeights) -> Result<()> {
        let key = (layer_id, expert_id);

        if self.slots.len() >= self.max_slots && !self.slots.contains_key(&key) {
            self.evict_lfu()?;
        }

        self.access_counter += 1;
        *self.last_access.entry(key).or_insert(0) = self.access_counter;

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

    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate()
    }

    pub fn resize(&mut self, new_max: usize) {
        let new_max = new_max.max(self.min_slots);
        while self.slots.len() > new_max {
            if self.evict_lfu().is_err() {
                break;
            }
        }
        self.max_slots = new_max;
    }

    fn evict_lfu(&mut self) -> Result<()> {
        if self.slots.is_empty() {
            return Err(anyhow!("no slots to evict"));
        }

        let mut min_freq = u64::MAX;
        let mut oldest_access = u64::MAX;
        let mut evict_key = (0usize, 0usize);

        for (&key, &freq) in &self.freq {
            if !self.slots.contains_key(&key) { continue; }
            let la = self.last_access.get(&key).copied().unwrap_or(0);
            if freq < min_freq || (freq == min_freq && la < oldest_access) {
                min_freq = freq;
                oldest_access = la;
                evict_key = key;
            }
        }

        self.slots.remove(&evict_key);
        self.freq.remove(&evict_key);
        self.last_access.remove(&evict_key);
        Ok(())
    }

    pub fn prefetch_slots(&self) -> usize {
        self.max_slots.saturating_sub(self.slots.len())
    }
}

pub struct RamExpertCache {
    hot: HashMap<(usize, usize), RamEntry>,
    cold: HashMap<(usize, usize), RamEntry>,
    hot_capacity: usize,
    cold_capacity: usize,
    hot_size: usize,
    cold_size: usize,
    min_hot_ratio: f64,
    max_hot_ratio: f64,
    stats: CacheHitStats,
}

struct RamEntry {
    data: Vec<u8>,
    freq: u64,
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
            min_hot_ratio: 0.2,
            max_hot_ratio: 0.6,
            stats: CacheHitStats::new(),
        }
    }

    pub fn get(&mut self, layer_id: usize, expert_id: usize) -> Option<Vec<u8>> {
        let key = (layer_id, expert_id);
        if let Some(entry) = self.hot.get_mut(&key) {
            entry.freq += 1;
            self.stats.record_hit();
            return Some(entry.data.clone());
        }
        if let Some(mut entry) = self.cold.remove(&key) {
            let size = entry.data.len();
            self.cold_size -= size;
            entry.freq += 1;
            while self.hot_size + size > self.hot_capacity && !self.hot.is_empty() {
                self.demote_hot();
            }
            self.hot.insert(key, entry);
            self.hot_size += size;
            self.stats.record_hit();
            return self.hot.get(&key).map(|e| e.data.clone());
        }
        self.stats.record_miss();
        None
    }

    pub fn put(&mut self, layer_id: usize, expert_id: usize, data: Vec<u8>) {
        let key = (layer_id, expert_id);
        let size = data.len();

        if self.hot.contains_key(&key) || self.cold.contains_key(&key) {
            return;
        }

        let entry = RamEntry { data, freq: 1 };
        if self.hot_size + size <= self.hot_capacity {
            self.hot.insert(key, entry);
            self.hot_size += size;
        } else if self.cold_size + size <= self.cold_capacity {
            self.cold.insert(key, entry);
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

    pub fn hit_rate(&self) -> f64 {
        self.stats.hit_rate()
    }

    pub fn rebalance(&mut self, hot_ratio: f64) {
        let hot_ratio = hot_ratio.clamp(self.min_hot_ratio, self.max_hot_ratio);
        let total_capacity = self.hot_capacity + self.cold_capacity;
        let new_hot = ((total_capacity as f64) * hot_ratio) as usize;
        let new_cold = total_capacity.saturating_sub(new_hot);

        while self.hot_size > new_hot && !self.hot.is_empty() {
            self.demote_hot();
        }

        self.hot_capacity = new_hot;
        self.cold_capacity = new_cold;
    }

    fn demote_hot(&mut self) {
        let mut min_freq = u64::MAX;
        let mut evict_key = None;
        for (&key, entry) in &self.hot {
            if entry.freq < min_freq {
                min_freq = entry.freq;
                evict_key = Some(key);
            }
        }
        if let Some(key) = evict_key {
            if let Some(entry) = self.hot.remove(&key) {
                let size = entry.data.len();
                self.hot_size -= size;
                if self.cold_size + size <= self.cold_capacity {
                    self.cold.insert(key, entry);
                    self.cold_size += size;
                }
            }
        }
    }
}

pub struct SsdExpertCache {
    base_path: String,
    index: HashMap<(usize, usize), u64>,
    stats: CacheHitStats,
}

impl SsdExpertCache {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            index: HashMap::new(),
            stats: CacheHitStats::new(),
        }
    }

    pub fn get(&mut self, layer_id: usize, expert_id: usize) -> Option<Vec<u8>> {
        let key = (layer_id, expert_id);
        if !self.index.contains_key(&key) {
            self.stats.record_miss();
            return None;
        }

        let path = self.expert_path(layer_id, expert_id);
        match std::fs::read(path) {
            Ok(data) => {
                self.stats.record_hit();
                Some(data)
            }
            Err(_) => {
                self.stats.record_miss();
                None
            }
        }
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

    pub fn prefetch_layers(&self, start_layer: usize, n_layers: usize, _n_experts: usize, expert_ids: &[usize]) -> Vec<(usize, usize, memmap2::Mmap)> {
        let mut results = Vec::new();
        for offset in 0..n_layers {
            let layer = start_layer + offset;
            for &eid in expert_ids {
                if let Some(mmap) = self.mmap_prefetch(layer, eid) {
                    results.push((layer, eid, mmap));
                }
            }
        }
        results
    }

    fn expert_path(&self, layer_id: usize, expert_id: usize) -> String {
        format!("{}/experts/{}_{}", self.base_path, layer_id, expert_id)
    }
}

pub struct ThreeLevelCache {
    pub gpu: GpuExpertCache,
    pub ram: RamExpertCache,
    pub ssd: SsdExpertCache,
    adapt_interval: u64,
    adapt_counter: u64,
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
            adapt_interval: 100,
            adapt_counter: 0,
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
            gpu_hit_rate: self.gpu.hit_rate(),
            ram_hot_entries: self.ram.hot_len(),
            ram_cold_entries: self.ram.cold_len(),
            ram_hit_rate: self.ram.hit_rate(),
            ssd_entries: self.ssd.index.len(),
        }
    }

    pub fn adapt(&mut self) {
        self.adapt_counter += 1;
        if self.adapt_counter < self.adapt_interval {
            return;
        }
        self.adapt_counter = 0;

        let gpu_hit = self.gpu.hit_rate();
        let ram_hit = self.ram.hit_rate();

        if gpu_hit > 0.9 {
            let new_max = (self.gpu.max_capacity() * 3 / 4).max(self.gpu.min_slots);
            self.gpu.resize(new_max);
        } else if gpu_hit < 0.5 {
            let new_max = (self.gpu.max_capacity() * 5 / 4).min(512);
            self.gpu.resize(new_max);
        }

        if ram_hit < 0.3 {
            self.ram.rebalance(0.4);
        } else if ram_hit > 0.8 {
            self.ram.rebalance(0.25);
        }
    }
}

pub struct CacheStats {
    pub gpu_slots_used: usize,
    pub gpu_slots_total: usize,
    pub gpu_hit_rate: f64,
    pub ram_hot_entries: usize,
    pub ram_cold_entries: usize,
    pub ram_hit_rate: f64,
    pub ssd_entries: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU[{}/{} hit:{:.1}%] RAM[hot:{} cold:{} hit:{:.1}%] SSD[{}]",
            self.gpu_slots_used, self.gpu_slots_total, self.gpu_hit_rate * 100.0,
            self.ram_hot_entries, self.ram_cold_entries, self.ram_hit_rate * 100.0,
            self.ssd_entries
        )
    }
}
