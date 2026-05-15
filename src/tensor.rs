use crate::dtype::DType;
use crate::pinned::PinnedBuffer;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use std::sync::Arc;

pub struct CpuTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl CpuTensor {
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::FP32 {
            return Err(anyhow!("expected FP32, got {}", self.dtype));
        }
        Ok(bytemuck::cast_slice(&self.data))
    }
}

pub struct GpuTensor {
    pub slice: CudaSlice<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Arc<CudaContext>,
}

impl Clone for GpuTensor {
    fn clone(&self) -> Self {
        let stream = self.device.default_stream();
        let nbytes = self.nbytes();
        let mut new_slice = stream
            .alloc_zeros::<u8>(nbytes)
            .expect("GPU alloc failed in clone");
        stream
            .memcpy_dtod(&self.slice, &mut new_slice)
            .expect("D2D copy failed in clone");
        Self {
            slice: new_slice,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}

impl GpuTensor {
    pub fn zeros(device: Arc<CudaContext>, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let nbytes = shape.iter().product::<usize>() * dtype.element_size();
        let stream = device.default_stream();
        let slice = stream
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| anyhow!("GPU alloc failed: {:?}", e))?;
        Ok(Self {
            slice,
            shape,
            dtype,
            device,
        })
    }

    pub fn from_host(device: Arc<CudaContext>, cpu: &CpuTensor) -> Result<Self> {
        let shape = cpu.shape.clone();
        let dtype = cpu.dtype;
        let stream = device.default_stream();
        let slice = stream
            .memcpy_stod(&cpu.data)
            .map_err(|e| anyhow!("H2D copy failed: {:?}", e))?;
        Ok(Self {
            slice,
            shape,
            dtype,
            device,
        })
    }

    pub fn to_host(&self) -> Result<CpuTensor> {
        let stream = self.device.default_stream();
        let host_data = stream
            .memcpy_dtov(&self.slice)
            .map_err(|e| anyhow!("D2H copy failed: {:?}", e))?;
        Ok(CpuTensor {
            data: host_data,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }

    pub fn device_ptr(&self) -> u64 {
        let stream = self.device.default_stream();
        let (ptr, _guard) = self.slice.device_ptr(&stream);
        ptr
    }

    pub fn strides(&self) -> Vec<usize> {
        let ndim = self.shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    pub fn as_dl_tensor(&self) -> crate::dlpack::DLTensor {
        let stream = self.device.default_stream();
        let (ptr, _guard) = self.slice.device_ptr(&stream);
        let device_id = self.device.ordinal() as i32;
        crate::dlpack::DLTensor::new(
            ptr,
            &self.shape,
            self.dtype,
            crate::dlpack::DLDevice::cuda(device_id),
        )
    }

    pub fn from_host_pinned(
        device: Arc<CudaContext>,
        cpu: &CpuTensor,
        pinned: &mut PinnedBuffer,
    ) -> Result<Self> {
        let nbytes = cpu.nbytes();
        pinned.copy_from(&cpu.data[..nbytes])?;

        let shape = cpu.shape.clone();
        let dtype = cpu.dtype;
        let stream = device.default_stream();
        let slice = stream
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| anyhow!("GPU alloc failed: {:?}", e))?;

        {
            let (dst_ptr, _guard) = slice.device_ptr(&stream);
            unsafe {
                cudarc::driver::sys::cuMemcpyAsync(
                    dst_ptr,
                    pinned.host_ptr(),
                    nbytes,
                    stream.cu_stream() as *mut _,
                );
            }
            stream.synchronize()?;
        }

        Ok(Self {
            slice,
            shape,
            dtype,
            device,
        })
    }

    pub fn from_host_pinned_async(
        device: Arc<CudaContext>,
        cpu: &CpuTensor,
        pinned: &mut PinnedBuffer,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<Self> {
        let nbytes = cpu.nbytes();
        pinned.copy_from(&cpu.data[..nbytes])?;

        let shape = cpu.shape.clone();
        let dtype = cpu.dtype;
        let default_stream = device.default_stream();
        let slice = default_stream
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| anyhow!("GPU alloc failed: {:?}", e))?;

        {
            let (dst_ptr, _guard) = slice.device_ptr(&stream);
            unsafe {
                cudarc::driver::sys::cuMemcpyAsync(
                    dst_ptr,
                    pinned.host_ptr(),
                    nbytes,
                    stream.cu_stream() as *mut _,
                );
            }
        }

        Ok(Self {
            slice,
            shape,
            dtype,
            device,
        })
    }

    pub fn gather_rows(&self, row_indices: &[usize], row_len: usize) -> Result<GpuTensor> {
        let n_rows = row_indices.len();
        let device = self.device.clone();
        let dst = GpuTensor::zeros(device.clone(), vec![n_rows, row_len], self.dtype)?;

        let row_bytes = row_len * self.dtype.element_size();

        let stream = device.default_stream();

        {
            let (src_ptr, _src_guard) = self.slice.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.slice.device_ptr(&stream);

            for (i, &src_row) in row_indices.iter().enumerate() {
                let src_offset = (src_row * row_bytes) as u64;
                let dst_offset = (i * row_bytes) as u64;
                unsafe {
                    cudarc::driver::sys::cuMemcpyAsync(
                        dst_ptr + dst_offset,
                        src_ptr + src_offset,
                        row_bytes,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            stream.synchronize()?;
        }

        Ok(dst)
    }

    pub fn scatter_add_rows(
        &mut self,
        src: &GpuTensor,
        dst_rows: &[usize],
        weights: &[f32],
        row_len: usize,
    ) -> Result<()> {
        let n_rows = dst_rows.len();
        let device = self.device.clone();
        let stream = device.default_stream();

        let src_host = src.to_host()?;
        let src_bf16: &[half::bf16] = bytemuck::cast_slice(&src_host.data);

        let mut self_host = self.to_host()?;
        let _self_bf16: &[half::bf16] = bytemuck::cast_slice(&self_host.data);
        let self_u16: &mut [u16] = bytemuck::cast_slice_mut(&mut self_host.data);

        for i in 0..n_rows {
            let w = weights[i];
            let dst_row = dst_rows[i];
            for d in 0..row_len {
                let cur = half::bf16::from_bits(self_u16[dst_row * row_len + d]).to_f32();
                let val = src_bf16[i * row_len + d].to_f32();
                self_u16[dst_row * row_len + d] =
                    half::bf16::from_f32(cur + w * val).to_bits();
            }
        }

        {
            let (dst_ptr, _guard) = self.slice.device_ptr(&stream);
            unsafe {
                cudarc::driver::sys::cuMemcpyAsync(
                    dst_ptr,
                    self_host.data.as_ptr() as u64,
                    self_host.data.len(),
                    stream.cu_stream() as *mut _,
                );
            }
            stream.synchronize()?;
        }

        Ok(())
    }

    pub fn copy_into_at(&self, dst: &mut GpuTensor, dst_offset_bytes: usize) -> Result<()> {
        let src_nbytes = self.nbytes();
        if dst_offset_bytes + src_nbytes > dst.nbytes() {
            return Err(anyhow!(
                "copy_into_at: src {} + offset {} > dst {}",
                src_nbytes, dst_offset_bytes, dst.nbytes()
            ));
        }
        let stream = self.device.default_stream();
        let (src_ptr, _src_guard) = self.slice.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.slice.device_ptr(&stream);
        unsafe {
            cudarc::driver::sys::cuMemcpyAsync(
                dst_ptr + dst_offset_bytes as u64,
                src_ptr,
                src_nbytes as usize,
                stream.cu_stream() as *mut _,
            );
        }
        stream.synchronize()?;
        Ok(())
    }

    pub fn d2d_scatter_rows(
        src: &GpuTensor,
        dst: &mut GpuTensor,
        src_batch_stride_bytes: usize,
        dst_batch_stride_bytes: usize,
        copy_bytes_per_batch: usize,
        dst_offset_bytes: usize,
        n_batches: usize,
    ) -> Result<()> {
        let stream = src.device.default_stream();
        let (src_ptr, _src_guard) = src.slice.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.slice.device_ptr(&stream);
        for b in 0..n_batches {
            let src_off = b * src_batch_stride_bytes;
            let dst_off = b * dst_batch_stride_bytes + dst_offset_bytes;
            if dst_off + copy_bytes_per_batch > dst.nbytes() {
                return Err(anyhow!(
                    "d2d_scatter_rows: batch {} dst_off {} + copy {} > dst {}",
                    b, dst_off, copy_bytes_per_batch, dst.nbytes()
                ));
            }
            unsafe {
                cudarc::driver::sys::cuMemcpyAsync(
                    dst_ptr + dst_off as u64,
                    src_ptr + src_off as u64,
                    copy_bytes_per_batch,
                    stream.cu_stream() as *mut _,
                );
            }
        }
        stream.synchronize()?;
        Ok(())
    }
}
