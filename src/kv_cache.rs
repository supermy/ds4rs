use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, DevicePtr};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

pub struct KvCache {
    pub cache: GpuTensor,
    pub window_size: usize,
    pub compress_ratio: usize,
    pub head_dim: usize,
    pub max_batch: usize,
    pub max_seqlen: usize,
    pub current_seqlen: HashMap<usize, usize>,
}

impl KvCache {
    pub fn new(
        device: Arc<CudaContext>,
        config: &ModelConfig,
        layer_id: usize,
        max_batch: usize,
        max_seqlen: usize,
    ) -> Result<Self> {
        let window_size = config.sliding_window;
        let compress_ratio = config.compress_ratio(layer_id) as usize;
        let head_dim = config.head_dim;
        let compressed_size = if compress_ratio > 0 { max_seqlen / compress_ratio } else { 0 };
        let total_size = window_size + compressed_size;

        let cache = GpuTensor::zeros(
            device,
            vec![max_batch, total_size, head_dim],
            DType::BF16,
        )?;

        Ok(Self {
            cache,
            window_size,
            compress_ratio,
            head_dim,
            max_batch,
            max_seqlen,
            current_seqlen: HashMap::new(),
        })
    }

    pub fn total_size(&self) -> usize {
        self.cache.shape[1]
    }

    pub fn compressed_size(&self) -> usize {
        if self.compress_ratio > 0 {
            self.max_seqlen / self.compress_ratio
        } else {
            0
        }
    }

    fn byte_offset(&self, batch_idx: usize, pos: usize) -> usize {
        let total_cols = self.cache.shape[1];
        (batch_idx * total_cols * self.head_dim + pos * self.head_dim) * 2
    }

    fn d2d_extract(&self, src_offset: usize, nbytes: usize, out_shape: Vec<usize>) -> Result<GpuTensor> {
        let device = self.cache.device.clone();
        let dst = GpuTensor::zeros(device, out_shape, DType::BF16)?;
        let stream = dst.device.default_stream();
        let (src_ptr, _src_guard) = self.cache.slice.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.slice.device_ptr(&stream);
        unsafe {
            cudarc::driver::sys::cuMemcpyAsync(
                dst_ptr,
                src_ptr + src_offset as u64,
                nbytes as usize,
                stream.cu_stream() as *mut _,
            );
        }
        drop(_src_guard);
        drop(_dst_guard);
        stream.synchronize()?;
        Ok(dst)
    }

    fn d2d_extract_rows(
        &self,
        src: &GpuTensor,
        row_start: usize,
        n_rows: usize,
    ) -> Result<GpuTensor> {
        let device = src.device.clone();
        let head_dim = self.head_dim;
        let elem_size = 2usize;
        let src_row_bytes = src.shape[1] * head_dim * elem_size;
        let _dst_row_bytes = n_rows * head_dim * elem_size;
        let src_col_offset = row_start * head_dim * elem_size;

        let out = GpuTensor::zeros(device.clone(), vec![1, n_rows, head_dim], DType::BF16)?;

        let stream = device.default_stream();
        {
            let (src_ptr, _src_guard) = src.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = out.slice.device_ptr(&stream);
            unsafe {
                cudarc::driver::sys::cuMemcpyAsync(
                    dst_ptr,
                    src_ptr + src_col_offset as u64,
                    (n_rows * head_dim * elem_size).min(src_row_bytes),
                    stream.cu_stream() as *mut _,
                );
            }
            stream.synchronize()?;
        }

        Ok(out)
    }

    pub fn update_prefill(
        &mut self,
        kv: &GpuTensor,
        batch_idx: usize,
        seqlen: usize,
    ) -> Result<()> {
        if batch_idx >= self.max_batch {
            return Err(anyhow!("batch index {} exceeds max batch size {}", batch_idx, self.max_batch));
        }

        let win = self.window_size;

        if seqlen <= win {
            let dst_offset = self.byte_offset(batch_idx, 0);
            kv.copy_into_at(&mut self.cache, dst_offset)?;
        } else {
            let cutoff = seqlen % win;
            let tail_start = seqlen - win;

            let tail_gpu = self.d2d_extract_rows(&kv, tail_start, win)?;

            if cutoff > 0 {
                let first_part = self.d2d_extract_rows(&tail_gpu, 0, win - cutoff)?;
                let dst_offset = self.byte_offset(batch_idx, cutoff);
                first_part.copy_into_at(&mut self.cache, dst_offset)?;

                let second_part = self.d2d_extract_rows(&tail_gpu, win - cutoff, cutoff)?;
                let head_dst = self.byte_offset(batch_idx, 0);
                second_part.copy_into_at(&mut self.cache, head_dst)?;
            } else {
                let dst_offset = self.byte_offset(batch_idx, 0);
                tail_gpu.copy_into_at(&mut self.cache, dst_offset)?;
            }
        }

        self.current_seqlen.insert(batch_idx, seqlen);
        Ok(())
    }

    pub fn update_decode(
        &mut self,
        kv: &GpuTensor,
        batch_idx: usize,
        start_pos: usize,
    ) -> Result<()> {
        if batch_idx >= self.max_batch {
            return Err(anyhow!("batch index {} exceeds max batch size {}", batch_idx, self.max_batch));
        }

        let pos = start_pos % self.window_size;
        let dst_offset = self.byte_offset(batch_idx, pos);
        kv.copy_into_at(&mut self.cache, dst_offset)?;

        let current = self.current_seqlen.entry(batch_idx).or_insert(0);
        *current = start_pos + 1;
        Ok(())
    }

    pub fn write_compressed(
        &mut self,
        compressed_kv: &GpuTensor,
        batch_idx: usize,
        start_pos: usize,
        _seqlen: usize,
    ) -> Result<()> {
        if self.compress_ratio == 0 {
            return Ok(());
        }

        let win = self.window_size;
        let comp_col = if start_pos == 0 { 0 } else { start_pos / self.compress_ratio };
        let dst_offset = self.byte_offset(batch_idx, win + comp_col);
        compressed_kv.copy_into_at(&mut self.cache, dst_offset)?;
        Ok(())
    }

    pub fn get_full_cache(&self, batch_idx: usize) -> Result<GpuTensor> {
        let current = self.current_seqlen.get(&batch_idx).copied().unwrap_or(0);
        let total_cols = self.cache.shape[1];
        let effective_len = if current <= self.window_size { current } else { total_cols };

        let src_offset = self.byte_offset(batch_idx, 0);
        let nbytes = effective_len * self.head_dim * 2;
        self.d2d_extract(src_offset, nbytes, vec![1, effective_len, self.head_dim])
    }

    pub fn get_window_kv(&self, batch_idx: usize, seqlen: usize, start_pos: usize) -> Result<GpuTensor> {
        let _ = (seqlen, start_pos);
        let current = self.current_seqlen.get(&batch_idx).copied().unwrap_or(0);
        let win = self.window_size;
        let actual_len = current.min(win);

        let src_offset = self.byte_offset(batch_idx, 0);
        let nbytes = actual_len * self.head_dim * 2;
        self.d2d_extract(src_offset, nbytes, vec![1, actual_len, self.head_dim])
    }

    pub fn get_compressed_kv(&self, batch_idx: usize, end_pos: usize) -> Result<GpuTensor> {
        if self.compress_ratio == 0 {
            return Err(anyhow!("no compressed KV for this layer"));
        }

        let win = self.window_size;
        let n_comp = end_pos / self.compress_ratio;
        if n_comp == 0 {
            return Err(anyhow!("no compressed KV available yet"));
        }

        let src_offset = self.byte_offset(batch_idx, win);
        let nbytes = n_comp * self.head_dim * 2;
        self.d2d_extract(src_offset, nbytes, vec![1, n_comp, self.head_dim])
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let host = self.cache.to_host()?;
        let total_cols = self.cache.shape[1];
        let data: &[u8] = &host.data;

        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        let bsz = self.max_batch as u64;
        let cols = total_cols as u64;
        let hd = self.head_dim as u64;
        let win = self.window_size as u64;
        let cr = self.compress_ratio as u64;
        let n_entries = self.current_seqlen.len() as u64;

        out.write_all(&bsz.to_le_bytes())?;
        out.write_all(&cols.to_le_bytes())?;
        out.write_all(&hd.to_le_bytes())?;
        out.write_all(&win.to_le_bytes())?;
        out.write_all(&cr.to_le_bytes())?;
        out.write_all(&n_entries.to_le_bytes())?;

        for (&batch_idx, &seqlen) in &self.current_seqlen {
            out.write_all(&(batch_idx as u64).to_le_bytes())?;
            out.write_all(&(seqlen as u64).to_le_bytes())?;
        }

        out.write_all(data)?;
        out.flush()?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        if data.len() < 48 {
            anyhow::bail!("checkpoint file too small");
        }

        let mut off = 0usize;
        let read_u64 = |data: &[u8], off: &mut usize| -> u64 {
            let v = u64::from_le_bytes(data[*off..*off + 8].try_into().unwrap());
            *off += 8;
            v
        };

        let _bsz = read_u64(&data, &mut off) as usize;
        let cols = read_u64(&data, &mut off) as usize;
        let hd = read_u64(&data, &mut off) as usize;
        let win = read_u64(&data, &mut off) as usize;
        let cr = read_u64(&data, &mut off) as usize;
        let n_entries = read_u64(&data, &mut off) as usize;

        if cols != self.cache.shape[1] || hd != self.head_dim || win != self.window_size || cr != self.compress_ratio {
            anyhow::bail!("checkpoint shape mismatch: expected cols={} hd={} win={} cr={}, got cols={} hd={} win={} cr={}",
                self.cache.shape[1], self.head_dim, self.window_size, self.compress_ratio,
                cols, hd, win, cr);
        }

        self.current_seqlen.clear();
        for _ in 0..n_entries {
            let batch_idx = read_u64(&data, &mut off) as usize;
            let seqlen = read_u64(&data, &mut off) as usize;
            self.current_seqlen.insert(batch_idx, seqlen);
        }

        let expected_bytes = self.max_batch * cols * hd * 2;
        if data.len() - off < expected_bytes {
            anyhow::bail!("checkpoint data truncated: expected {} bytes, got {}", expected_bytes, data.len() - off);
        }

        let cpu = crate::tensor::CpuTensor::new(
            data[off..off + expected_bytes].to_vec(),
            vec![self.max_batch, cols, hd],
            DType::BF16,
        );
        self.cache = GpuTensor::from_host(self.cache.device.clone(), &cpu)?;

        Ok(())
    }
}
