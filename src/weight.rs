use crate::dtype::DType;
use crate::tensor::CpuTensor;
use anyhow::{anyhow, Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

enum CachedFile {
    Vec(Vec<u8>),
    Mmap(memmap2::Mmap),
}

impl CachedFile {
    fn as_slice(&self) -> &[u8] {
        match self {
            CachedFile::Vec(v) => v,
            CachedFile::Mmap(m) => m,
        }
    }
}

pub struct WeightLoader {
    model_dir: PathBuf,
    weight_map: HashMap<String, String>,
    file_cache: HashMap<String, CachedFile>,
    use_mmap: bool,
}

impl WeightLoader {
    pub fn from_dir(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);
        let index_path = dir.join("model.safetensors.index.json");

        let weight_map = if index_path.exists() {
            let content = fs::read_to_string(&index_path)
                .with_context(|| format!("failed to read index from {:?}", index_path))?;
            let index: serde_json::Value = serde_json::from_str(&content)
                .with_context(|| "failed to parse safetensors index")?;
            index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect::<HashMap<String, String>>()
                })
                .unwrap_or_default()
        } else {
            Self::build_weight_map_from_files(dir)?
        };

        if weight_map.is_empty() {
            return Err(anyhow!("no safetensors weights found in {:?}", model_dir));
        }

        Ok(Self {
            model_dir: dir.to_path_buf(),
            weight_map,
            file_cache: HashMap::new(),
            use_mmap: true,
        })
    }

    pub fn from_dir_no_mmap(model_dir: &str) -> Result<Self> {
        let mut loader = Self::from_dir(model_dir)?;
        loader.use_mmap = false;
        Ok(loader)
    }

    fn build_weight_map_from_files(dir: &Path) -> Result<HashMap<String, String>> {
        let mut weight_map = HashMap::new();
        for entry in fs::read_dir(dir).with_context(|| format!("failed to read dir {:?}", dir))? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("model-") || !name.ends_with(".safetensors") {
                continue;
            }
            let filepath = dir.join(&name);
            let data = fs::read(&filepath)
                .with_context(|| format!("failed to read {:?}", filepath))?;
            let st = SafeTensors::deserialize(&data)
                .with_context(|| format!("failed to parse {:?}", filepath))?;
            for tensor_name in st.names() {
                weight_map.insert(tensor_name.to_string(), name.clone());
            }
        }
        Ok(weight_map)
    }

    pub fn list_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.weight_map.keys().map(|s| s.as_str()).collect();
        keys.sort();
        keys
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn file_for(&self, name: &str) -> Option<&str> {
        self.weight_map.get(name).map(|s| s.as_str())
    }

    fn ensure_file(&mut self, filename: &str) -> Result<()> {
        if self.file_cache.contains_key(filename) {
            return Ok(());
        }
        let filepath = self.model_dir.join(filename);

        if self.use_mmap {
            let file = fs::File::open(&filepath)
                .with_context(|| format!("failed to open {:?}", filepath))?;
            let mmap = unsafe { memmap2::Mmap::map(&file)
                .with_context(|| format!("failed to mmap {:?}", filepath))? };
            self.file_cache.insert(filename.to_string(), CachedFile::Mmap(mmap));
        } else {
            let data = fs::read(&filepath)
                .with_context(|| format!("failed to read {:?}", filepath))?;
            self.file_cache.insert(filename.to_string(), CachedFile::Vec(data));
        }
        Ok(())
    }

    fn deserialize_cached(&self, filename: &str) -> Result<SafeTensors<'_>> {
        let cached = self.file_cache.get(filename)
            .ok_or_else(|| anyhow!("file {} not cached", filename))?;
        SafeTensors::deserialize(cached.as_slice())
            .with_context(|| format!("failed to parse safetensors from {}", filename))
    }

    fn collect_keys_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.weight_map.keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }

    fn load_keys(&mut self, keys: &[String]) -> Result<HashMap<String, CpuTensor>> {
        let mut result = HashMap::new();
        let mut filenames: Vec<String> = Vec::new();
        for key in keys {
            let filename = self.weight_map.get(key.as_str())
                .ok_or_else(|| anyhow!("weight {} not found", key))?
                .clone();
            if !filenames.contains(&filename) {
                filenames.push(filename);
            }
        }
        for filename in &filenames {
            self.ensure_file(filename)?;
        }
        for key in keys {
            let filename = self.weight_map.get(key.as_str()).unwrap();
            let st = self.deserialize_cached(filename)?;
            let tensor = st.tensor(key)
                .with_context(|| format!("tensor {} not found in {}", key, filename))?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let dtype = Self::safetensors_dtype_to_dtype(tensor.dtype())
                .with_context(|| format!("unsupported dtype for {}", key))?;
            let data = tensor.data().to_vec();
            result.insert(key.clone(), CpuTensor::new(data, shape, dtype));
        }
        Ok(result)
    }

    pub fn load(&mut self, name: &str) -> Result<CpuTensor> {
        let filename = self.weight_map.get(name)
            .ok_or_else(|| anyhow!("weight {} not found in weight_map", name))?
            .clone();
        self.ensure_file(&filename)?;
        let st = self.deserialize_cached(&filename)?;
        let tensor = st.tensor(name)
            .with_context(|| format!("tensor {} not found in {}", name, filename))?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = Self::safetensors_dtype_to_dtype(tensor.dtype())
            .with_context(|| format!("unsupported dtype for tensor {}", name))?;
        let data = tensor.data().to_vec();
        Ok(CpuTensor::new(data, shape, dtype))
    }

    pub fn load_layer(&mut self, layer_id: usize) -> Result<HashMap<String, CpuTensor>> {
        let prefix = format!("layers.{}.", layer_id);
        let keys = self.collect_keys_with_prefix(&prefix);
        self.load_keys(&keys)
    }

    pub fn load_expert(
        &mut self,
        layer_id: usize,
        expert_id: usize,
    ) -> Result<HashMap<String, CpuTensor>> {
        let prefix = format!("layers.{}.ffn.experts.{}.", layer_id, expert_id);
        let keys = self.collect_keys_with_prefix(&prefix);
        if keys.is_empty() {
            return Err(anyhow!("no weights for layer {} expert {}", layer_id, expert_id));
        }
        self.load_keys(&keys)
    }

    pub fn load_shared_experts(&mut self, layer_id: usize) -> Result<HashMap<String, CpuTensor>> {
        let prefix = format!("layers.{}.ffn.shared_experts.", layer_id);
        let keys = self.collect_keys_with_prefix(&prefix);
        self.load_keys(&keys)
    }

    pub fn load_non_moe(&mut self) -> Result<HashMap<String, CpuTensor>> {
        let keys: Vec<String> = self.weight_map.keys()
            .filter(|k| {
                if k.starts_with("layers.") && k.contains(".ffn.experts.") {
                    return false;
                }
                if k.starts_with("layers.") && k.contains(".ffn.shared_experts.") {
                    return false;
                }
                true
            })
            .cloned()
            .collect();
        self.load_keys(&keys)
    }

    pub fn layer_expert_keys(&self, layer_id: usize, expert_id: usize) -> Vec<String> {
        let prefix = format!("layers.{}.ffn.experts.{}.", layer_id, expert_id);
        self.collect_keys_with_prefix(&prefix)
    }

    pub fn layer_shared_expert_keys(&self, layer_id: usize) -> Vec<String> {
        let prefix = format!("layers.{}.ffn.shared_experts.", layer_id);
        self.collect_keys_with_prefix(&prefix)
    }

    pub fn layer_attn_keys(&self, layer_id: usize) -> Vec<String> {
        let prefix = format!("layers.{}.attn.", layer_id);
        let norm_prefix = format!("layers.{}.attn_norm.", layer_id);
        let ffn_norm_prefix = format!("layers.{}.ffn_norm.", layer_id);
        self.weight_map.keys()
            .filter(|k| k.starts_with(&prefix) || k.starts_with(&norm_prefix) || k.starts_with(&ffn_norm_prefix))
            .cloned()
            .collect()
    }

    pub fn evict_file(&mut self, filename: &str) {
        self.file_cache.remove(filename);
    }

    pub fn evict_all(&mut self) {
        self.file_cache.clear();
    }

    pub fn cached_file_count(&self) -> usize {
        self.file_cache.len()
    }

    fn safetensors_dtype_to_dtype(dt: safetensors::Dtype) -> Result<DType> {
        match dt {
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::F32 => Ok(DType::FP32),
            safetensors::Dtype::F8_E4M3 => Ok(DType::FP8E4M3),
            safetensors::Dtype::F8_E8M0 => Ok(DType::FP8E8M0),
            safetensors::Dtype::U8 => Ok(DType::UINT8),
            safetensors::Dtype::I8 => Ok(DType::UINT8),
            safetensors::Dtype::I32 => Ok(DType::INT32),
            safetensors::Dtype::I64 => Ok(DType::INT64),
            other => Err(anyhow!("unsupported safetensors dtype: {:?}", other)),
        }
    }
}
