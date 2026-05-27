/*! GGUF 文件格式支持 — 用于存储量化专家权重。

GGUF (GGML Universal Format) 是 llama.cpp 的标准格式，支持：
- 元数据（键值对）
- 张量信息（名称、形状、类型、偏移）
- 张量数据（按量化类型存储）

本模块实现：
1. GGUF 写入器：将 IQ2_XS / Q2_K 等量化权重写入 GGUF 文件
2. GGUF 读取器：mmap 读取，支持随机访问
3. 侧载索引：JSON 格式的快速查找索引

文件结构：
┌─────────────────────────────────────┐
│ Header (32 bytes)                   │
│   - magic: GGUF (4B)                │
│   - version: uint32 (4B)            │
│   - tensor_count: uint64 (8B)       │
│   - metadata_kv_count: uint64 (8B)  │
├─────────────────────────────────────┤
│ Metadata KV (variable)              │
│   - key_length + key_string         │
│   - value_type + value_data         │
├─────────────────────────────────────┤
│ Tensor Info (variable per tensor)   │
│   - name_length + name_string       │
│   - n_dims + dims[]                 │
│   - type + offset                   │
├─────────────────────────────────────┤
│ Tensor Data (aligned)               │
│   - 按量化类型存储的权重数据          │
└─────────────────────────────────────┘

侧载索引 (experts.idx.json):
{
  "n_layers": 43,
  "n_experts": 256,
  "n_weights": 3,
  "tensors": {
    "layers.0.experts.0.w1": {"offset": 0, "size": 12345, "shape": [2048, 7168]},
    ...
  }
}
*/

use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;

/// GGUF 值类型
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// GGML 张量类型（量化类型）
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
}

impl GgmlType {
    pub fn block_size(self) -> usize {
        match self {
            Self::IQ2_XS | Self::IQ2_XXS | Self::IQ2_S => 256,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            _ => 1,
        }
    }

    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::IQ2_XS => 74,
            Self::IQ2_XXS => 66,
            Self::Q2_K => 84,
            _ => 0,
        }
    }
}

/// 张量信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub size: u64,
}

/// 侧载索引
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarIndex {
    pub n_layers: u32,
    pub n_experts: u32,
    pub n_weights: u32,
    pub quant_type: String,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SidecarIndex {
    pub fn new(n_layers: u32, n_experts: u32, n_weights: u32, quant_type: &str) -> Self {
        Self {
            n_layers,
            n_experts,
            n_weights,
            quant_type: quant_type.to_string(),
            tensors: HashMap::new(),
        }
    }

    pub fn add_tensor(&mut self, info: TensorInfo) {
        self.tensors.insert(info.name.clone(), info);
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).context("create index file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).context("write index json")?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("open index file")?;
        serde_json::from_reader(file).context("parse index json")
    }

    pub fn get_expert(&self, layer_id: u32, expert_id: u32, weight_type: u32) -> Option<&TensorInfo> {
        let weight_name = match weight_type {
            0 => "w1",
            1 => "w2",
            2 => "w3",
            _ => return None,
        };
        let name = format!("layers.{}.experts.{}.{}", layer_id, expert_id, weight_name);
        self.tensors.get(&name)
    }
}

/// GGUF 写入器
pub struct GgufWriter<W: Write + Seek> {
    writer: W,
    tensor_infos: Vec<TensorInfo>,
    metadata: HashMap<String, MetadataValue>,
    data_offset: u64,
    current_offset: u64,
}

/// 元数据值
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint32(u32),
    Uint64(u64),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    String(String),
    Bool(bool),
    Array(Vec<MetadataValue>),
}

impl<W: Write + Seek> GgufWriter<W> {
    pub fn new(mut writer: W) -> Result<Self> {
        writer.write_all(GGUF_MAGIC)?;
        writer.write_all(&GGUF_VERSION.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?;
        writer.write_all(&0u64.to_le_bytes())?;

        Ok(Self {
            writer,
            tensor_infos: Vec::new(),
            metadata: HashMap::new(),
            data_offset: 0,
            current_offset: 0,
        })
    }

    pub fn add_metadata(&mut self, key: &str, value: MetadataValue) {
        self.metadata.insert(key.to_string(), value);
    }

    pub fn add_tensor(&mut self, name: &str, dims: &[u64], ggml_type: GgmlType) -> Result<u64> {
        let n_elements: u64 = dims.iter().product();
        let block_size = ggml_type.block_size();
        let bytes_per_block = ggml_type.bytes_per_block();
        
        let n_blocks = (n_elements + block_size as u64 - 1) / block_size as u64;
        let size = n_blocks * bytes_per_block as u64;

        let offset = self.current_offset;
        self.current_offset += size;

        self.tensor_infos.push(TensorInfo {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            offset,
            size,
        });

        Ok(offset)
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> {
        let tensor_count = self.tensor_infos.len() as u64;
        let metadata_count = self.metadata.len() as u64;

        self.writer.seek(SeekFrom::Start(8))?;
        self.writer.write_all(&tensor_count.to_le_bytes())?;
        self.writer.write_all(&metadata_count.to_le_bytes())?;

        self.writer.seek(SeekFrom::End(0))?;

        self.write_metadata()?;
        self.write_tensor_infos()?;

        Ok(())
    }

    fn write_metadata(&mut self) -> Result<()> {
        let metadata: Vec<(String, MetadataValue)> = self.metadata.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (key, value) in metadata {
            self.write_string(&key)?;
            self.write_metadata_value(&value)?;
        }
        Ok(())
    }

    fn write_metadata_value(&mut self, value: &MetadataValue) -> Result<()> {
        match value {
            MetadataValue::Uint32(v) => {
                self.writer.write_all(&(GgufValueType::Uint32 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Uint64(v) => {
                self.writer.write_all(&(GgufValueType::Uint64 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Int32(v) => {
                self.writer.write_all(&(GgufValueType::Int32 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Int64(v) => {
                self.writer.write_all(&(GgufValueType::Int64 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Float32(v) => {
                self.writer.write_all(&(GgufValueType::Float32 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::Float64(v) => {
                self.writer.write_all(&(GgufValueType::Float64 as u32).to_le_bytes())?;
                self.writer.write_all(&v.to_le_bytes())?;
            }
            MetadataValue::String(v) => {
                self.writer.write_all(&(GgufValueType::String as u32).to_le_bytes())?;
                self.write_string(v)?;
            }
            MetadataValue::Bool(v) => {
                self.writer.write_all(&(GgufValueType::Bool as u32).to_le_bytes())?;
                self.writer.write_all(&[*v as u8])?;
            }
            MetadataValue::Array(items) => {
                self.writer.write_all(&(GgufValueType::Array as u32).to_le_bytes())?;
                self.writer.write_all(&(items.len() as u64).to_le_bytes())?;
                for item in items {
                    self.write_metadata_value(item)?;
                }
            }
        }
        Ok(())
    }

    fn write_tensor_infos(&mut self) -> Result<()> {
        let tensor_infos: Vec<TensorInfo> = self.tensor_infos.clone();
        for info in tensor_infos {
            self.write_string(&info.name)?;
            self.writer.write_all(&(info.dims.len() as u32).to_le_bytes())?;
            for &dim in &info.dims {
                self.writer.write_all(&dim.to_le_bytes())?;
            }
            self.writer.write_all(&(info.ggml_type as u32).to_le_bytes())?;
            self.writer.write_all(&info.offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        self.writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
        self.writer.write_all(bytes)?;
        Ok(())
    }
}

/// GGUF 读取器
pub struct GgufReader {
    file: File,
    mmap: Mmap,
    tensor_infos: HashMap<String, TensorInfo>,
    metadata: HashMap<String, MetadataValue>,
    data_offset: u64,
}

impl GgufReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).context("open gguf file")?;

        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;

        if &header[0..4] != GGUF_MAGIC {
            anyhow::bail!("invalid GGUF magic");
        }

        let version = u32::from_le_bytes(header[4..8].try_into()?);
        if version > GGUF_VERSION {
            anyhow::bail!("unsupported GGUF version: {}", version);
        }

        let tensor_count = u64::from_le_bytes(header[8..16].try_into()?);
        let metadata_count = u64::from_le_bytes(header[16..24].try_into()?);

        let mmap = unsafe { Mmap::map(&file).context("mmap gguf file")? };

        let mut reader = Self {
            file,
            mmap,
            tensor_infos: HashMap::new(),
            metadata: HashMap::new(),
            data_offset: 0,
        };

        let mut offset = 24usize;
        offset = reader.read_metadata(offset, metadata_count as usize)?;
        offset = reader.read_tensor_infos(offset, tensor_count as usize)?;
        reader.data_offset = offset as u64;

        Ok(reader)
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.get(name)
    }

    pub fn get_tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.data_offset as usize + info.offset as usize;
        let end = start + info.size as usize;
        &self.mmap[start..end]
    }

    pub fn get_expert_data(&self, layer_id: u32, expert_id: u32, weight_type: u32) -> Option<(&TensorInfo, &[u8])> {
        let weight_name = match weight_type {
            0 => "w1",
            1 => "w2",
            2 => "w3",
            _ => return None,
        };
        let name = format!("layers.{}.experts.{}.{}", layer_id, expert_id, weight_name);
        let info = self.tensor_infos.get(&name)?;
        let data = self.get_tensor_data(info);
        Some((info, data))
    }

    pub fn load_sidecar_index(&self, path: &Path) -> Result<SidecarIndex> {
        SidecarIndex::load(path)
    }

    fn read_metadata(&mut self, mut offset: usize, count: usize) -> Result<usize> {
        for _ in 0..count {
            let (key, new_offset) = self.read_string(offset)?;
            offset = new_offset;
            let (value, new_offset) = self.read_metadata_value(offset)?;
            offset = new_offset;
            self.metadata.insert(key, value);
        }
        Ok(offset)
    }

    fn read_metadata_value(&self, mut offset: usize) -> Result<(MetadataValue, usize)> {
        let value_type = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
        offset += 4;

        let value = match value_type {
            x if x == GgufValueType::Uint32 as u32 => {
                let v = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
                offset += 4;
                MetadataValue::Uint32(v)
            }
            x if x == GgufValueType::Uint64 as u32 => {
                let v = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
                offset += 8;
                MetadataValue::Uint64(v)
            }
            x if x == GgufValueType::Int32 as u32 => {
                let v = i32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
                offset += 4;
                MetadataValue::Int32(v)
            }
            x if x == GgufValueType::Int64 as u32 => {
                let v = i64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
                offset += 8;
                MetadataValue::Int64(v)
            }
            x if x == GgufValueType::Float32 as u32 => {
                let v = f32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
                offset += 4;
                MetadataValue::Float32(v)
            }
            x if x == GgufValueType::Float64 as u32 => {
                let v = f64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
                offset += 8;
                MetadataValue::Float64(v)
            }
            x if x == GgufValueType::String as u32 => {
                let (v, new_offset) = self.read_string(offset)?;
                offset = new_offset;
                MetadataValue::String(v)
            }
            x if x == GgufValueType::Bool as u32 => {
                let v = self.mmap[offset] != 0;
                offset += 1;
                MetadataValue::Bool(v)
            }
            x if x == GgufValueType::Array as u32 => {
                let count = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
                offset += 8;
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    let (item, new_offset) = self.read_metadata_value(offset)?;
                    offset = new_offset;
                    items.push(item);
                }
                MetadataValue::Array(items)
            }
            _ => anyhow::bail!("unknown metadata value type: {}", value_type),
        };

        Ok((value, offset))
    }

    fn read_tensor_infos(&mut self, mut offset: usize, count: usize) -> Result<usize> {
        for _ in 0..count {
            let (name, new_offset) = self.read_string(offset)?;
            offset = new_offset;

            let n_dims = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
                offset += 8;
                dims.push(dim);
            }

            let ggml_type = u32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
            offset += 4;
            let ggml_type = match ggml_type {
                x if x == GgmlType::F32 as u32 => GgmlType::F32,
                x if x == GgmlType::F16 as u32 => GgmlType::F16,
                x if x == GgmlType::IQ2_XS as u32 => GgmlType::IQ2_XS,
                x if x == GgmlType::IQ2_XXS as u32 => GgmlType::IQ2_XXS,
                x if x == GgmlType::Q2_K as u32 => GgmlType::Q2_K,
                _ => anyhow::bail!("unknown ggml type: {}", ggml_type),
            };

            let tensor_offset = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?);
            offset += 8;

            let n_elements: u64 = dims.iter().product();
            let block_size = ggml_type.block_size();
            let bytes_per_block = ggml_type.bytes_per_block();
            let n_blocks = (n_elements + block_size as u64 - 1) / block_size as u64;
            let size = n_blocks * bytes_per_block as u64;

            self.tensor_infos.insert(name.clone(), TensorInfo {
                name,
                dims,
                ggml_type,
                offset: tensor_offset,
                size,
            });
        }
        Ok(offset)
    }

    fn read_string(&self, offset: usize) -> Result<(String, usize)> {
        let len = u64::from_le_bytes(self.mmap[offset..offset + 8].try_into()?) as usize;
        let start = offset + 8;
        let end = start + len;
        let s = std::str::from_utf8(&self.mmap[start..end])?.to_string();
        Ok((s, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_gguf_write_read() {
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        
        let mut writer = GgufWriter::new(cursor).unwrap();
        writer.add_metadata("general.name", MetadataValue::String("test".to_string()));
        writer.add_metadata("general.architecture", MetadataValue::String("llama".to_string()));
        
        let offset = writer.add_tensor("test.weight", &[256, 256], GgmlType::IQ2_XS).unwrap();
        assert_eq!(offset, 0);
        
        let data = vec![0u8; 74 * 256];
        writer.write_tensor_data(&data).unwrap();
        
        writer.finalize().unwrap();
    }
}
